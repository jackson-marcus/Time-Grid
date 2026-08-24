"""API routes: /leaderboard, /forecast, /whatif, /hierarchy, /health."""

from __future__ import annotations

import functools
import logging
import pickle

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from timegrid.models.reconcile import coherence_error, hierarchy, reconcile
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


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/hierarchy")
def get_hierarchy() -> dict:
    h = hierarchy()
    return {"total": "total", "regions": h["middle"], "bottom": h["bottom"]}


@router.get("/leaderboard")
def leaderboard() -> dict:
    try:
        bundle = _bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"metrics": bundle["metrics"]}


@router.get("/forecast")
def forecast(method: str = "ols") -> dict:
    if method not in {"base", "bottom_up", "top_down", "ols"}:
        raise HTTPException(status_code=422, detail="method must be base|bottom_up|top_down|ols")
    try:
        bundle = _bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    forecasts = reconcile(bundle["base"], method, bundle["shares"])
    weeks = [bundle["last_week"] + 1 + i for i in range(len(forecasts["total"]))]
    return {
        "method": method,
        "weeks": weeks,
        "coherence_error": coherence_error(forecasts),
        "forecasts": {name: np.round(values, 1).tolist() for name, values in forecasts.items()},
        "history_tail": bundle["history_tail"],
    }


@router.post("/whatif")
def whatif(request: WhatIfRequest) -> dict:
    try:
        bundle = _bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    h = hierarchy()
    if request.series not in h["bottom"]:
        raise HTTPException(status_code=404, detail=f"series must be a bottom node: {h['bottom']}")

    # baseline and scenario must use the SAME reconciliation (bottom-up), or the
    # delta would mostly measure the difference between reconciliation methods
    baseline = reconcile(bundle["base"], "bottom_up", bundle["shares"])
    adjusted_base = {k: v.copy() for k, v in bundle["base"].items()}
    bump = np.ones_like(adjusted_base[request.series])
    bump[: request.weeks] += request.uplift_pct / 100
    adjusted_base[request.series] = adjusted_base[request.series] * bump
    adjusted = reconcile(adjusted_base, "bottom_up", bundle["shares"])

    def delta(name: str) -> float:
        return round(float((adjusted[name] - baseline[name]).sum()), 1)

    region = request.series.split("/")[0]
    return {
        "scenario": request.model_dump(),
        "impact": {
            request.series: delta(request.series),
            region: delta(region),
            "total": delta("total"),
        },
        "coherence_error": coherence_error(adjusted),
    }
