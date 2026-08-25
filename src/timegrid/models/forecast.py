"""Per-series base forecasts: ridge on trend + Fourier features.

The models are deliberately independent — one fit per node of the hierarchy —
which is exactly why their outputs need reconciling afterwards. Which series
exist, and how they roll up, is the tree's business, not this module's.

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

from timegrid.hierarchy import Node, build_hierarchy, reconcile
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


def base_forecasts(df: pd.DataFrame, horizon: int, root: Node) -> dict[str, np.ndarray]:
    """One independent fit per node of the hierarchy, root and leaves alike."""
    cfg = get_config()["forecast"]
    t = df["week"].to_numpy()
    return {
        node.name: fit_and_forecast(df[node.name].to_numpy(), t, horizon, cfg["ridge_alpha"])
        for node in root.level_order()
    }


def history_shares(df: pd.DataFrame, root: Node) -> dict[str, float]:
    """Each leaf's historical fraction of the root — the top-down allocation key."""
    total = df[root.name].sum()
    return {leaf.name: float(df[leaf.name].sum() / total) for leaf in root.leaves()}


def backtest() -> dict:
    cfg = get_config()
    root = build_hierarchy()
    df = pd.read_parquet(resolve_path(cfg["data"]["processed_dir"]) / "kpis.parquet")
    holdout = cfg["forecast"]["holdout_weeks"]
    train_df, test_df = df.iloc[:-holdout], df.iloc[-holdout:]

    base = base_forecasts(train_df, holdout, root)
    shares = history_shares(train_df, root)

    # score each method at each depth of the tree
    levels = [
        ("total", [root]),
        ("region", list(root.children)),
        ("bottom", list(root.leaves())),
    ]

    metrics = {}
    for method in ["base", "bottom_up", "top_down", "ols"]:
        forecasts = reconcile(root, base, method, shares)
        for level_name, nodes in levels:
            errors = []
            for node in nodes:
                actual = test_df[node.name].to_numpy()
                errors.append(np.mean(np.abs(forecasts[node.name] - actual) / actual))
            metrics[f"mape_{level_name}_{method}"] = round(float(np.mean(errors)), 4)
        metrics[f"coherence_{method}"] = round(root.coherence_error(forecasts), 4)

    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(cfg["eval"]["experiment_name"])
    with mlflow.start_run(run_name="reconciliation"):
        mlflow.log_params({"holdout": holdout})
        mlflow.log_metrics(metrics)
    logger.info("reconciliation %s", metrics)

    # production forecasts from the full history
    full_base = base_forecasts(df, cfg["forecast"]["horizon"], root)
    full_shares = history_shares(df, root)
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
