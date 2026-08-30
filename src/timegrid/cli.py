"""Typer CLI entry point for the kpi_series batch pipeline."""

from __future__ import annotations

import typer

from timegrid.pipeline.dag import topological_order
from timegrid.pipeline.executor import PipelineExecutor

app = typer.Typer(help="Run the kpi_series batch DAG.")


@app.command("plan")
def plan() -> None:
    typer.echo(" -> ".join(topological_order()))


@app.command("run")
def run() -> None:
    result = PipelineExecutor().run([])
    typer.echo(result["registered_model"]["name"])


def main() -> None:
    app()
