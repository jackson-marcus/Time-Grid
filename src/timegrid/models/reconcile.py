"""Hierarchy structure + forecast reconciliation (bottom-up, top-down, OLS)."""

from __future__ import annotations

import numpy as np

from timegrid.settings import get_config


def hierarchy() -> dict:
    cfg = get_config()["data"]
    bottom = [f"{r}/{p}" for r in cfg["regions"] for p in cfg["products"]]
    return {
        "bottom": bottom,
        "middle": list(cfg["regions"]),
        "all_series": ["total", *cfg["regions"], *bottom],
    }


def summing_matrix() -> tuple[np.ndarray, list[str]]:
    """S maps bottom-level values to every node: rows [total, regions..., bottom...]."""
    cfg = get_config()["data"]
    h = hierarchy()
    n_bottom = len(h["bottom"])
    rows = [np.ones(n_bottom)]  # total
    for region in cfg["regions"]:
        rows.append(np.array([1.0 if b.startswith(f"{region}/") else 0.0 for b in h["bottom"]]))
    rows.extend(np.eye(n_bottom))
    return np.vstack(rows), h["all_series"]


def reconcile(
    base: dict[str, np.ndarray], method: str, history_share: dict[str, float]
) -> dict[str, np.ndarray]:
    """base: series -> forecast vector. Returns coherent forecasts for all nodes."""
    s_matrix, order = summing_matrix()
    h = hierarchy()
    horizon = len(next(iter(base.values())))

    if method == "base":
        return base
    if method == "bottom_up":
        bottom = np.vstack([base[b] for b in h["bottom"]])
    elif method == "top_down":
        shares = np.array([history_share[b] for b in h["bottom"]])
        bottom = np.outer(shares, np.ones(horizon)) * base["total"]
    elif method == "ols":
        # y_tilde = S (S'S)^-1 S' y_hat — orthogonal projection onto coherent space
        y_hat = np.vstack([base[name] for name in order])
        projection = s_matrix @ np.linalg.solve(s_matrix.T @ s_matrix, s_matrix.T @ y_hat)
        return {name: projection[i] for i, name in enumerate(order)}
    else:
        raise ValueError(method)

    stacked = s_matrix @ bottom
    return {name: stacked[i] for i, name in enumerate(order)}


def coherence_error(forecasts: dict[str, np.ndarray]) -> float:
    """Max absolute violation of the summing constraints."""
    cfg = get_config()["data"]
    h = hierarchy()
    errors = [np.abs(forecasts["total"] - sum(forecasts[r] for r in cfg["regions"])).max()]
    for region in cfg["regions"]:
        children = sum(forecasts[f"{region}/{p}"] for p in cfg["products"])
        errors.append(np.abs(forecasts[region] - children).max())
    return float(max(errors))
