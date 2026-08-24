"""Offline fixtures: generated KPI hierarchy backtested into tmp artifacts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from make_kpis import generate  # noqa: E402

from timegrid.settings import get_config, get_settings  # noqa: E402


@pytest.fixture(scope="session")
def kpis():
    cfg = get_config()["data"]
    return generate(140, cfg["regions"], cfg["products"], seed=7)


@pytest.fixture(scope="session")
def backtested(tmp_path_factory, kpis):
    tmp = tmp_path_factory.mktemp("timegrid")
    (tmp / "processed").mkdir()
    kpis.to_parquet(tmp / "processed" / "kpis.parquet", index=False)

    cfg = get_config()
    originals = (cfg["data"]["processed_dir"], cfg["data"]["artifacts_dir"])
    cfg["data"]["processed_dir"] = str(tmp / "processed")
    cfg["data"]["artifacts_dir"] = str(tmp / "artifacts")

    old_uri = os.environ.get("MLFLOW_TRACKING_URI")
    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{tmp / 'mlflow.db'}"
    get_settings.cache_clear()

    from timegrid.models.forecast import backtest

    metrics = backtest()
    yield {"metrics": metrics, "artifacts": tmp / "artifacts"}

    cfg["data"]["processed_dir"], cfg["data"]["artifacts_dir"] = originals
    if old_uri is None:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    else:
        os.environ["MLFLOW_TRACKING_URI"] = old_uri
    get_settings.cache_clear()


@pytest.fixture
def api_client(backtested):
    from fastapi.testclient import TestClient

    from timegrid.api import routes
    from timegrid.api.main import app

    routes._bundle.cache_clear()
    try:
        yield TestClient(app)
    finally:
        routes._bundle.cache_clear()
