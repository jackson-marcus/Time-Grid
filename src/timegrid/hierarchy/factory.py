"""The one place that turns configuration into a hierarchy.

Everything downstream receives a :class:`~timegrid.hierarchy.node.Node` and asks
the tree its questions. No other module reads ``regions``/``products`` from the
config, and no other module composes or splits a ``"region/product"`` string.
"""

from __future__ import annotations

import functools

from timegrid.hierarchy.node import GroupNode, LeafNode, Node
from timegrid.settings import get_config

ROOT_NAME = "total"
LEAF_SEPARATOR = "/"


def leaf_name(region: str, product: str) -> str:
    """The canonical column/series name of a bottom node."""
    return f"{region}{LEAF_SEPARATOR}{product}"


def compose(root_name: str, regions: list[str], products: list[str]) -> Node:
    """Build a two-level total -> region -> product composite."""
    return GroupNode(
        root_name,
        [
            GroupNode(region, [LeafNode(leaf_name(region, product)) for product in products])
            for region in regions
        ],
    )


@functools.lru_cache(maxsize=1)
def build_hierarchy() -> Node:
    """The project's KPI tree, built once from ``configs/config.yaml``."""
    cfg = get_config()["data"]
    return compose(ROOT_NAME, list(cfg["regions"]), list(cfg["products"]))
