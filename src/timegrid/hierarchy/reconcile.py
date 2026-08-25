"""Reconciliation expressed as traversals of the composite.

Independent per-series base forecasts do not add up. Each method below makes them
coherent by a different movement through the same tree:

    bottom_up  post-order fold      — believe the leaves, sum upward
    top_down   pre-order allocation — believe the root, split downward by share
    ols        orthogonal projection onto the tree's column space
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from timegrid.hierarchy.node import Node, SeriesMap
from timegrid.hierarchy.summing import build_summing_matrix

METHODS = ("base", "bottom_up", "top_down", "ols")


def bottom_up(root: Node, base: SeriesMap) -> dict[str, np.ndarray]:
    """Post-order fold: each group becomes the sum of its children."""
    return root.fold(base)


def top_down(root: Node, base: SeriesMap, shares: Mapping[str, float]) -> dict[str, np.ndarray]:
    """Pre-order allocation: the root's forecast cascades down by historical share."""
    return root.distribute(np.asarray(base[root.name], dtype=float), shares)


def ols(root: Node, base: SeriesMap) -> dict[str, np.ndarray]:
    """y~ = S (S'S)^-1 S' y^ — project the base forecasts onto the coherent subspace."""
    s_matrix, order = build_summing_matrix(root)
    y_hat = np.vstack([np.asarray(base[name], dtype=float) for name in order])
    projection = s_matrix @ np.linalg.solve(s_matrix.T @ s_matrix, s_matrix.T @ y_hat)
    return {name: projection[i] for i, name in enumerate(order)}


def reconcile(
    root: Node,
    base: SeriesMap,
    method: str,
    shares: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Coherent forecasts for every node, keyed in the tree's level order."""
    if method == "base":
        reconciled = {name: np.asarray(base[name], dtype=float) for name in base}
    elif method == "bottom_up":
        reconciled = bottom_up(root, base)
    elif method == "top_down":
        if shares is None:
            raise ValueError("top_down reconciliation needs historical leaf shares")
        reconciled = top_down(root, base, shares)
    elif method == "ols":
        reconciled = ols(root, base)
    else:
        raise ValueError(method)
    return {node.name: reconciled[node.name] for node in root.level_order()}


def coherence_error(root: Node, forecasts: SeriesMap) -> float:
    """Worst summing-constraint violation anywhere in the tree."""
    return root.coherence_error(forecasts)
