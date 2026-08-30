"""`validate` step for kpi_series."""

from __future__ import annotations

from typing import Any


def run(context: dict[str, Any]) -> dict[str, Any]:
    rows = list(context.get("rows", []))
    context = dict(context)
    context["validate"] = {
        "n_rows": len(rows),
        "entity": "kpi_series",
        "ok": True,
    }
    context.setdefault("trace", []).append("validate")
    return context
