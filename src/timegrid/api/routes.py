"""API routes: /leaderboard, /forecast, /whatif, /hierarchy, /health.

Every route works from the composite tree returned by ``build_hierarchy()``:
node names come from nodes, the region affected by a what-if comes from the
leaf's ancestors, and coherence is asked of the root.
"""

from __future__ import annotations

import functools
import logging
import pickle

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from timegrid.hierarchy import METHODS, build_hierarchy, reconcile
from timegrid.settings import get_config, resolve_path

logger = logging.getLogger(__name__)
router = APIRouter()


class WhatIfRequest(BaseModel):
    series: str
    uplift_pct: float = Field(ge=-50, le=100)
    weeks: int = Field(ge=1, le=12, default=8)


@functools.lru_cache(maxsize=1)
def _bundle() -> dict:
    path = resolve_path(get_config()["data"]["artifacts_dir"]) / "bundle.pkl"
    if not path.exists():
        raise FileNotFoundError("Artifacts missing; run make_kpis.py + timegrid.models.forecast")
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_bundle() -> dict:
    try:
        return _bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/hierarchy")
def get_hierarchy() -> dict:
    root = build_hierarchy()
    return {
        "total": root.name,
        "regions": [child.name for child in root.children],
        "bottom": [leaf.name for leaf in root.leaves()],
    }


@router.get("/leaderboard")
def leaderboard() -> dict:
    return {"metrics": _load_bundle()["metrics"]}


@router.get("/forecast")
def forecast(method: str = "ols") -> dict:
    if method not in METHODS:
        raise HTTPException(status_code=422, detail="method must be base|bottom_up|top_down|ols")
    bundle = _load_bundle()
    root = build_hierarchy()
    forecasts = reconcile(root, bundle["base"], method, bundle["shares"])
    weeks = [bundle["last_week"] + 1 + i for i in range(len(forecasts[root.name]))]
    return {
        "method": method,
        "weeks": weeks,
        "coherence_error": root.coherence_error(forecasts),
        "forecasts": {name: np.round(values, 1).tolist() for name, values in forecasts.items()},
        "history_tail": bundle["history_tail"],
    }


@router.post("/whatif")
def whatif(request: WhatIfRequest) -> dict:
    bundle = _load_bundle()
    root = build_hierarchy()
    leaf_names = [leaf.name for leaf in root.leaves()]
    if request.series not in leaf_names:
        raise HTTPException(status_code=404, detail=f"series must be a bottom node: {leaf_names}")
    leaf = root.find(request.series)

    # baseline and scenario must use the SAME reconciliation (bottom-up), or the
    # delta would mostly measure the difference between reconciliation methods
    baseline = reconcile(root, bundle["base"], "bottom_up")
    adjusted_base = {k: v.copy() for k, v in bundle["base"].items()}
    bump = np.ones_like(adjusted_base[leaf.name])
    bump[: request.weeks] += request.uplift_pct / 100
    adjusted_base[leaf.name] = adjusted_base[leaf.name] * bump
    adjusted = reconcile(root, adjusted_base, "bottom_up")

    def delta(name: str) -> float:
        return round(float((adjusted[name] - baseline[name]).sum()), 1)

    # the intervention is felt exactly by the leaf and everything that contains it
    impact = {node.name: delta(node.name) for node in (leaf, *leaf.ancestors())}
    return {
        "scenario": request.model_dump(),
        "impact": impact,
        "coherence_error": root.coherence_error(adjusted),
    }
