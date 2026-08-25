"""Composite hierarchy, reconciliation traversals, leaderboard, what-if, API."""

from __future__ import annotations

import numpy as np
import pytest

from timegrid.hierarchy import (
    GroupNode,
    LeafNode,
    build_hierarchy,
    build_summing_matrix,
    compose,
    reconcile,
)

# --------------------------------------------------------------- the composite


@pytest.fixture(scope="module")
def root():
    return build_hierarchy()


def test_composite_shape_and_uniform_interface(root):
    assert root.name == "total"
    assert not root.is_leaf
    assert [child.name for child in root.children] == ["north", "south", "west"]
    assert [leaf.name for leaf in root.leaves()] == [
        f"{r}/{p}" for r in ["north", "south", "west"] for p in ["alpha", "beta", "gamma"]
    ]
    assert len(root) == 13  # total + 3 regions + 9 bottom

    # every node answers the same questions, leaf or group, with no type checks
    for node in root.walk():
        assert isinstance(node.leaves(), tuple)
        assert node.is_leaf == (len(node.children) == 0)
        assert node.share_weight(dict.fromkeys([n.name for n in root.leaves()], 1.0)) == len(
            node.leaves()
        )

    leaf = root.find("north/alpha")
    assert isinstance(leaf, LeafNode)
    assert leaf.leaves() == (leaf,)
    assert [a.name for a in leaf.ancestors()] == ["north", "total"]
    assert isinstance(root.find("north"), GroupNode)
    assert "west/gamma" in root
    with pytest.raises(KeyError):
        root.find("nope")


def test_traversal_orders_differ_as_documented(root):
    pre = [n.name for n in root.walk()]
    level = [n.name for n in root.level_order()]
    assert pre[:3] == ["total", "north", "north/alpha"]  # depth-first
    assert level[:5] == ["total", "north", "south", "west", "north/alpha"]  # breadth-first
    assert sorted(pre) == sorted(level)


def test_fold_is_a_post_order_aggregation(root):
    rng = np.random.default_rng(1)
    leaf_values = {leaf.name: rng.uniform(10, 100, 4) for leaf in root.leaves()}
    folded = root.fold(leaf_values)

    assert set(folded) == {node.name for node in root.walk()}
    for leaf in root.leaves():
        assert np.allclose(folded[leaf.name], leaf_values[leaf.name])
    for group in (root, *root.children):
        assert np.allclose(folded[group.name], sum(folded[c.name] for c in group.children))
    assert np.allclose(root.aggregate(leaf_values), sum(leaf_values.values()))
    # a subtree folds on its own, with no reference to the root
    assert np.allclose(root.find("south").aggregate(leaf_values), folded["south"])


def test_distribute_is_a_pre_order_share_allocation(root):
    leaves = [leaf.name for leaf in root.leaves()]
    shares = dict.fromkeys(leaves, 1 / len(leaves))
    shares["north/alpha"] = 2 / (len(leaves) + 1)  # one leaf twice as big
    for name in leaves[1:]:
        shares[name] = 1 / (len(leaves) + 1)

    allocated = root.distribute(np.array([1200.0, 1200.0]), shares)
    assert np.allclose(allocated["total"], 1200.0)  # the root keeps what it was given
    assert np.allclose(allocated["north/alpha"], 2 * allocated["north/beta"])
    for group in (root, *root.children):  # allocation is coherent by construction
        assert np.allclose(allocated[group.name], sum(allocated[c.name] for c in group.children))
    assert root.coherence_error(allocated) < 1e-9


def test_coherence_error_is_a_recursive_tree_invariant(root):
    coherent = root.fold({leaf.name: np.full(3, 10.0) for leaf in root.leaves()})
    assert root.coherence_error(coherent) == 0.0
    assert root.find("north/beta").coherence_error(coherent) == 0.0  # leaves never disagree

    broken = dict(coherent)
    broken["south"] = broken["south"] + 7.0  # a violation two levels down
    assert root.coherence_error(broken) == pytest.approx(7.0)
    assert root.find("south").coherence_error(broken) == pytest.approx(7.0)
    assert root.find("north").coherence_error(broken) == 0.0  # a sibling stays clean


def test_summing_matrix_is_derived_from_the_tree(root):
    s_matrix, order = build_summing_matrix(root)
    assert s_matrix.shape == (13, 9)  # total + 3 regions + 9 bottom
    assert order == [node.name for node in root.level_order()]
    assert order[0] == "total"

    leaves = [leaf.name for leaf in root.leaves()]
    for row, name in enumerate(order):
        beneath = {leaf.name for leaf in root.find(name).leaves()}
        assert list(s_matrix[row]) == [1.0 if leaf in beneath else 0.0 for leaf in leaves]

    # S is a flattening of the tree: S @ bottom reproduces the post-order fold
    rng = np.random.default_rng(3)
    bottom = rng.uniform(10, 100, (9, 4))
    stacked = s_matrix @ bottom
    folded = root.fold(dict(zip(leaves, bottom, strict=True)))
    for i, name in enumerate(order):
        assert np.allclose(stacked[i], folded[name])


def test_summing_matrix_follows_a_different_tree_shape():
    """Nothing is hard-coded to total/region/product — the matrix is whatever the tree is."""
    tiny = compose("total", ["north"], ["alpha", "beta"])
    s_matrix, order = build_summing_matrix(tiny)
    assert order == ["total", "north", "north/alpha", "north/beta"]
    assert s_matrix.tolist() == [[1, 1], [1, 1], [1, 0], [0, 1]]

    deep = GroupNode("total", [GroupNode("north", [LeafNode("north/alpha")]), LeafNode("direct")])
    s_deep, order_deep = build_summing_matrix(deep)
    assert order_deep == ["total", "north", "direct", "north/alpha"]
    assert s_deep.shape == (4, 2)  # leaves at mixed depths, no special casing


# ------------------------------------------------------- reconciliation & data


def test_generator_hierarchy_is_exactly_coherent(kpis, root):
    observed = {node.name: kpis[node.name].to_numpy() for node in root.walk()}
    assert root.coherence_error(observed) < 0.5


def test_ols_projection_is_idempotent_on_coherent_input(root):
    s_matrix, order = build_summing_matrix(root)
    rng = np.random.default_rng(0)
    bottom = rng.uniform(10, 100, (9, 4))
    coherent = {name: (s_matrix @ bottom)[i] for i, name in enumerate(order)}
    projected = reconcile(root, coherent, "ols")
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


def test_reconcile_rejects_unknown_method_and_missing_shares(root):
    base = {node.name: np.ones(3) for node in root.walk()}
    with pytest.raises(ValueError, match="magic"):
        reconcile(root, base, "magic")
    with pytest.raises(ValueError, match="shares"):
        reconcile(root, base, "top_down")


# ----------------------------------------------------------------------- HTTP


def test_api_hierarchy_is_the_tree(api_client):
    body = api_client.get("/hierarchy").json()
    assert body["total"] == "total"
    assert body["regions"] == ["north", "south", "west"]
    assert len(body["bottom"]) == 9


def test_api_forecast_and_whatif(api_client):
    assert api_client.get("/health").json() == {"status": "ok"}

    body = api_client.get("/forecast", params={"method": "ols"}).json()
    assert body["coherence_error"] < 1e-6
    assert len(body["forecasts"]["total"]) == 12
    assert api_client.get("/forecast", params={"method": "magic"}).status_code == 422

    scenario = api_client.post(
        "/whatif", json={"series": "north/alpha", "uplift_pct": 10, "weeks": 8}
    ).json()
    assert list(scenario["impact"]) == ["north/alpha", "north", "total"]  # leaf then its ancestors
    assert scenario["impact"]["north/alpha"] > 0
    assert (
        abs(scenario["impact"]["total"] - scenario["impact"]["north/alpha"]) < 1.0
    )  # only one series changed; totals move by the same amount
    assert api_client.post("/whatif", json={"series": "total", "uplift_pct": 10}).status_code == 404
