"""Per-series base forecasts: ridge on trend + Fourier features.

Usage (backtest + fit):
    python -m timegrid.models.forecast
"""

from __future__ import annotations

import logging
import pickle

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from timegrid.models.reconcile import coherence_error, hierarchy, reconcile
from timegrid.settings import get_config, get_settings, resolve_path

logger = logging.getLogger(__name__)


def _features(t: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            t,
            np.sin(2 * np.pi * t / 52),
            np.cos(2 * np.pi * t / 52),
            np.sin(4 * np.pi * t / 52),
            np.cos(4 * np.pi * t / 52),
        ]
    )


def fit_and_forecast(series: np.ndarray, t: np.ndarray, horizon: int, alpha: float) -> np.ndarray:
    """Ridge in LOG space (multiplicative trend/seasonality for positive KPIs).

    The log transform is also what makes base forecasts genuinely incoherent:
    a purely linear fit would be additive (forecast of a sum = sum of
    forecasts), and reconciliation would have nothing to fix."""
    model = Ridge(alpha=alpha)
    model.fit(_features(t), np.log(series))
    future = np.arange(t[-1] + 1, t[-1] + 1 + horizon)
    return np.exp(model.predict(_features(future)))


def base_forecasts(df: pd.DataFrame, horizon: int) -> dict[str, np.ndarray]:
    cfg = get_config()["forecast"]
    h = hierarchy()
    t = df["week"].to_numpy()
    return {
        name: fit_and_forecast(df[name].to_numpy(), t, horizon, cfg["ridge_alpha"])
        for name in h["all_series"]
    }


def history_shares(df: pd.DataFrame) -> dict[str, float]:
    h = hierarchy()
    total = df["total"].sum()
    return {b: float(df[b].sum() / total) for b in h["bottom"]}


def backtest() -> dict:
    cfg = get_config()
    df = pd.read_parquet(resolve_path(cfg["data"]["processed_dir"]) / "kpis.parquet")
    holdout = cfg["forecast"]["holdout_weeks"]
    train_df, test_df = df.iloc[:-holdout], df.iloc[-holdout:]

    base = base_forecasts(train_df, holdout)
    shares = history_shares(train_df)
    h = hierarchy()

    metrics = {}
    per_method = {}
    for method in ["base", "bottom_up", "top_down", "ols"]:
        forecasts = reconcile(base, method, shares)
        per_method[method] = forecasts
        for level_name, series_names in [
            ("total", ["total"]),
            ("region", h["middle"]),
            ("bottom", h["bottom"]),
        ]:
            errors = []
            for name in series_names:
                actual = test_df[name].to_numpy()
                errors.append(np.mean(np.abs(forecasts[name] - actual) / actual))
            metrics[f"mape_{level_name}_{method}"] = round(float(np.mean(errors)), 4)
        metrics[f"coherence_{method}"] = round(coherence_error(forecasts), 4)

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(cfg["eval"]["experiment_name"])
    with mlflow.start_run(run_name="reconciliation"):
        mlflow.log_params({"holdout": holdout})
        mlflow.log_metrics(metrics)
    logger.info("reconciliation %s", metrics)

    # production forecasts from the full history
    full_base = base_forecasts(df, cfg["forecast"]["horizon"])
    full_shares = history_shares(df)
    artifacts = resolve_path(cfg["data"]["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    with open(artifacts / "bundle.pkl", "wb") as f:
        pickle.dump(
            {
                "metrics": metrics,
                "base": full_base,
                "shares": full_shares,
                "last_week": int(df["week"].max()),
                "history_tail": df.tail(52).to_dict(orient="list"),
            },
            f,
        )
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    backtest()
