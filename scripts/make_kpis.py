"""Hierarchical weekly KPIs: total -> regions -> region x product bottom series.

Bottom series carry distinct trends, seasonality phases, promo bumps, and
noise; aggregates are exact sums, so coherence is a testable invariant.

Usage:
    uv run python scripts/make_kpis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from timegrid.hierarchy import leaf_name
from timegrid.settings import get_config, resolve_path


def generate(n_weeks: int, regions: list[str], products: list[str], seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_weeks)
    frames = {}
    for region in regions:
        for product in products:
            base = rng.uniform(400, 1600)
            trend = rng.uniform(-1.0, 3.0) * t
            phase = rng.uniform(0, 2 * np.pi)
            season = rng.uniform(0.08, 0.22) * base * np.sin(2 * np.pi * t / 52 + phase)
            promo = (rng.random(n_weeks) < 0.06) * rng.uniform(0.2, 0.5) * base
            noise = rng.normal(0, 0.05 * base, n_weeks)
            frames[leaf_name(region, product)] = np.maximum(
                base + trend + season + promo + noise, 10
            )

    df = pd.DataFrame({"week": t})
    for name, series in frames.items():
        df[name] = np.round(series, 1)
    for region in regions:
        df[region] = df[[leaf_name(region, p) for p in products]].sum(axis=1).round(1)
    df["total"] = df[regions].sum(axis=1).round(1)
    return df


def main() -> None:
    cfg = get_config()["data"]
    df = generate(cfg["n_weeks"], cfg["regions"], cfg["products"], cfg["seed"])
    out = resolve_path(cfg["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "kpis.parquet", index=False)
    print(json.dumps({"weeks": len(df), "series": len(df.columns) - 1}))


if __name__ == "__main__":
    main()
