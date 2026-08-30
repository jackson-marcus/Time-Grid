"""Ordered batch steps for kpi_series."""

from . import evaluate, features, ingest, register, train, validate

STEP_RUNNERS = {
    "ingest": ingest.run,
    "validate": validate.run,
    "features": features.run,
    "train": train.run,
    "evaluate": evaluate.run,
    "register": register.run,
}
