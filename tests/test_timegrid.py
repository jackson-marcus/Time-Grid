"""Hierarchy coherence, reconciliation math, leaderboard, what-if, API."""

from __future__ import annotations

import numpy as np

from timegrid.models.reconcile import hierarchy, reconcile, summing_matrix


def test_generator_hierarchy_is_exactly_coherent(kpis):
    cfg_regions = ["north", "south", "west"]
    assert np.allclose(kpis["total"], kpis[cfg_regions].sum(axis=1), atol=0.5)
    assert np.allclose(
        kpis["north"],
        kpis[[c for c in kpis.columns if c.startswith("north/")]].sum(axis=1),
        atol=0.5,
    )


def test_summing_matrix_shape_and_projection_idempotence():
    s_matrix, order = summing_matrix()
    assert s_matrix.shape == (13, 9)  # total + 3 regions + 9 bottom
    assert order[0] == "total"

    rng = np.random.default_rng(0)
    bottom = rng.uniform(10, 100, (9, 4))
    coherent = {name: (s_matrix @ bottom)[i] for i, name in enumerate(order)}
    projected = reconcile(coherent, "ols", dict.fromkeys(hierarchy()["bottom"], 1 / 9))
    for name in order:  # projecting an already-coherent forecast changes nothing
        assert np.allclose(projected[name], coherent[name], atol=1e-8)


def test_reconciliation_restores_coherence(backtested):
    m = backtested["metrics"]
    assert m["coherence_base"] > 1.0  # independent base models disagree
    for method in ["bottom_up", "top_down", "ols"]:
        assert m[f"coherence_{method}"] < 1e-6


def test_leaderboard_is_honest(backtested):
    m = backtested["metrics"]
    # OLS reconciliation should not hurt the top level materially
    assert m["mape_total_ols"] <= m["mape_total_base"] + 0.01
    # top-down destroys bottom-level accuracy relative to bottom-up (classic)
    assert m["mape_bottom_top_down"] >= m["mape_bottom_bottom_up"]
    assert all(0.001 < v < 0.5 for k, v in m.items() if k.startswith("mape_"))


def test_api_forecast_and_whatif(api_client):
    assert api_client.get("/health").json() == {"status": "ok"}

    body = api_client.get("/forecast", params={"method": "ols"}).json()
    assert body["coherence_error"] < 1e-6
    assert len(body["forecasts"]["total"]) == 12
    assert api_client.get("/forecast", params={"method": "magic"}).status_code == 422

    scenario = api_client.post(
        "/whatif", json={"series": "north/alpha", "uplift_pct": 10, "weeks": 8}
    ).json()
    assert scenario["impact"]["north/alpha"] > 0
    assert (
        abs(scenario["impact"]["total"] - scenario["impact"]["north/alpha"]) < 1.0
    )  # only one series changed; totals move by the same amount
    assert api_client.post("/whatif", json={"series": "total", "uplift_pct": 10}).status_code == 404
