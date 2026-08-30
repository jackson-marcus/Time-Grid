"""DAG for the kpi_series batch pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepSpec:
    name: str
    requires: tuple[str, ...]


DAG_STEPS: tuple[StepSpec, ...] = (
    StepSpec("ingest", ()),
    StepSpec("validate", ("ingest",)),
    StepSpec("features", ("validate",)),
    StepSpec("train", ("features",)),
    StepSpec("evaluate", ("train",)),
    StepSpec("register", ("evaluate",)),
)


def predecessors() -> dict[str, tuple[str, ...]]:
    return {step.name: step.requires for step in DAG_STEPS}


def topological_order() -> list[str]:
    remaining = {step.name: set(step.requires) for step in DAG_STEPS}
    ordered: list[str] = []
    while remaining:
        ready = sorted(name for name, deps in remaining.items() if not deps)
        if not ready:
            raise RuntimeError("cycle in batch DAG")
        name = ready[0]
        ordered.append(name)
        remaining.pop(name)
        for deps in remaining.values():
            deps.discard(name)
    return ordered
