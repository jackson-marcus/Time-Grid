"""The KPI hierarchy as a Composite: the tree *is* the object graph.

A hierarchical forecasting problem is a tree of business units — a total that is
the sum of regions, each of which is the sum of its products. Every operation the
system performs (aggregating, checking coherence, distributing shares, building a
summing matrix) is a recursion over that tree, so the tree is modelled directly:

    Node                  abstract member of the hierarchy
    ├── LeafNode          a bottom series that owns observed values
    └── GroupNode         a series *defined* as the sum of its children

Both implement the same protocol, so callers never ask "is this a region or a
product?" — they call ``fold`` / ``distribute`` / ``coherence_error`` on whatever
node they hold and the recursion terminates itself at the leaves.

The three recursions correspond exactly to the three reconciliation moves:

    fold()             post-order  — bottom-up aggregation
    distribute()       pre-order   — top-down share allocation
    coherence_error()  post-order  — the summing invariant as a tree property
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator, Mapping, Sequence

import numpy as np

SeriesMap = Mapping[str, np.ndarray]
"""Series name -> value vector (a forecast horizon, or a column of history)."""


class Node(ABC):
    """A member of the KPI hierarchy. Leaves and groups share this interface."""

    __slots__ = ("_name", "_parent")

    def __init__(self, name: str) -> None:
        self._name = name
        self._parent: Node | None = None

    # ------------------------------------------------------------------ shape

    @property
    def name(self) -> str:
        return self._name

    @property
    def parent(self) -> Node | None:
        return self._parent

    def attach_to(self, parent: Node) -> None:
        """Called once by the enclosing group at construction; a node has one home."""
        if self._parent is not None:
            raise ValueError(f"{self._name!r} already belongs to {self._parent.name!r}")
        self._parent = parent

    @property
    @abstractmethod
    def children(self) -> tuple[Node, ...]:
        """Direct children — empty for a leaf."""

    @property
    def is_leaf(self) -> bool:
        return not self.children

    # -------------------------------------------------------------- traversal

    def walk(self) -> Iterator[Node]:
        """Depth-first pre-order: this node, then each subtree."""
        yield self
        for child in self.children:
            yield from child.walk()

    def level_order(self) -> Iterator[Node]:
        """Breadth-first: total, then regions, then bottom series.

        This is the canonical row order of the summing matrix and of every
        reconciled forecast dict the API returns.
        """
        queue: deque[Node] = deque([self])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children)

    def leaves(self) -> tuple[Node, ...]:
        """Bottom series under this node, left to right (a leaf is its own leaf)."""
        return tuple(node for node in self.walk() if node.is_leaf)

    def ancestors(self) -> tuple[Node, ...]:
        """Enclosing nodes, nearest first, ending at the root."""
        chain: list[Node] = []
        node = self._parent
        while node is not None:
            chain.append(node)
            node = node.parent
        return tuple(chain)

    def find(self, name: str) -> Node:
        """The node called ``name`` in this subtree."""
        for node in self.walk():
            if node.name == name:
                return node
        raise KeyError(name)

    def __contains__(self, name: object) -> bool:
        return any(node.name == name for node in self.walk())

    def __len__(self) -> int:
        return sum(1 for _ in self.walk())

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._name!r}, children={len(self.children)})"

    # ------------------------------------------------------------- recursions

    @abstractmethod
    def fold(self, series_map: SeriesMap) -> dict[str, np.ndarray]:
        """Post-order fold: every node's value, computed from its leaves upward."""

    @abstractmethod
    def distribute(
        self, value: np.ndarray, leaf_shares: Mapping[str, float]
    ) -> dict[str, np.ndarray]:
        """Pre-order allocation: split ``value`` down the tree by subtree share."""

    @abstractmethod
    def share_weight(self, leaf_shares: Mapping[str, float]) -> float:
        """Total historical share of the leaves beneath this node."""

    @abstractmethod
    def coherence_error(self, series_map: SeriesMap) -> float:
        """Worst violation of "a group equals the sum of its children" in this subtree."""

    def aggregate(self, series_map: SeriesMap) -> np.ndarray:
        """This node's value implied by the leaves beneath it."""
        return self.fold(series_map)[self._name]


class LeafNode(Node):
    """A bottom series (region x product). It owns values; it derives nothing."""

    __slots__ = ()

    @property
    def children(self) -> tuple[Node, ...]:
        return ()

    def fold(self, series_map: SeriesMap) -> dict[str, np.ndarray]:
        return {self._name: np.asarray(series_map[self._name], dtype=float)}

    def distribute(
        self, value: np.ndarray, leaf_shares: Mapping[str, float]
    ) -> dict[str, np.ndarray]:
        return {self._name: value}

    def share_weight(self, leaf_shares: Mapping[str, float]) -> float:
        return float(leaf_shares[self._name])

    def coherence_error(self, series_map: SeriesMap) -> float:
        return 0.0  # a leaf has no children to disagree with


class GroupNode(Node):
    """A series defined as the sum of its children (the total, or one region)."""

    __slots__ = ("_children",)

    def __init__(self, name: str, children: Sequence[Node]) -> None:
        super().__init__(name)
        if not children:
            raise ValueError(f"group node {name!r} needs at least one child")
        self._children = tuple(children)
        for child in self._children:
            child.attach_to(self)

    @property
    def children(self) -> tuple[Node, ...]:
        return self._children

    def fold(self, series_map: SeriesMap) -> dict[str, np.ndarray]:
        folded: dict[str, np.ndarray] = {}
        for child in self._children:  # children first ...
            folded.update(child.fold(series_map))
        folded[self._name] = sum(folded[child.name] for child in self._children)  # ... then us
        return folded

    def distribute(
        self, value: np.ndarray, leaf_shares: Mapping[str, float]
    ) -> dict[str, np.ndarray]:
        allocated: dict[str, np.ndarray] = {self._name: value}  # us first ...
        weights = [child.share_weight(leaf_shares) for child in self._children]
        pool = sum(weights)
        if pool <= 0:
            raise ValueError(f"children of {self._name!r} carry no historical share")
        for child, weight in zip(self._children, weights, strict=True):
            # ... then each subtree gets its relative slice of what we were given
            allocated.update(child.distribute(value * (weight / pool), leaf_shares))
        return allocated

    def share_weight(self, leaf_shares: Mapping[str, float]) -> float:
        return sum(child.share_weight(leaf_shares) for child in self._children)

    def coherence_error(self, series_map: SeriesMap) -> float:
        children_sum = sum(
            np.asarray(series_map[child.name], dtype=float) for child in self._children
        )
        here = float(np.abs(np.asarray(series_map[self._name], dtype=float) - children_sum).max())
        return max(here, *(child.coherence_error(series_map) for child in self._children))
