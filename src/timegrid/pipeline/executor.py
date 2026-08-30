"""Execute the kpi_series DAG and record wrmsfe."""

from __future__ import annotations

from typing import Any

from timegrid.pipeline.dag import topological_order
from timegrid.pipeline.steps import STEP_RUNNERS


class PipelineExecutor:
    def run(self, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        context: dict[str, Any] = {"rows": list(rows or []), "trace": [], "metric": "wrmsfe"}
        for name in topological_order():
            context = STEP_RUNNERS[name](context)
        context["registered_model"] = {
            "name": "kpi_series",
            "target": "value",
            "metric": "wrmsfe",
        }
        return context
