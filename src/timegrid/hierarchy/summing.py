"""The summing matrix, derived by walking the composite.

S is not a primary object here — it is a flattening of the tree, produced on
demand for the one method (OLS projection) that needs linear algebra rather than
recursion. Row order is the tree's level order; column order is its leaf order.
"""

from __future__ import annotations

import numpy as np

from timegrid.hierarchy.node import Node


def build_summing_matrix(root: Node) -> tuple[np.ndarray, list[str]]:
    """S[node, leaf] = 1 iff that leaf sits beneath that node.

    Returns the matrix and the node names in row order.
    """
    leaves = root.leaves()
    column = {leaf.name: i for i, leaf in enumerate(leaves)}
    nodes = list(root.level_order())

    s_matrix = np.zeros((len(nodes), len(leaves)))
    for row, node in enumerate(nodes):
        for leaf in node.leaves():  # the composite answers this for leaves and groups alike
            s_matrix[row, column[leaf.name]] = 1.0
    return s_matrix, [node.name for node in nodes]
