"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI

from timegrid import __version__
from timegrid.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    app = FastAPI(
        title="timegrid",
        description="Hierarchical KPI forecasting: per-series Fourier-ridge base forecasts, bottom-up, top-down, and OLS-projection reconciliation with a coherence-and-accuracy leaderboard, plus a coherent what-if planner.",
        version=__version__,
    )
    app.include_router(router)
    return app


app = create_app()
