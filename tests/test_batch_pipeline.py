"""Pattern #5 — Raw → ETL → Features → Train → Evaluate → Registry."""

from timegrid.cli import app
from timegrid.pipeline.dag import DAG_STEPS, topological_order
from timegrid.pipeline.executor import PipelineExecutor


def test_dag_is_acyclic_and_complete():
    order = topological_order()
    assert order == ["ingest", "validate", "features", "train", "evaluate", "register"]
    assert len(DAG_STEPS) == 6


def test_executor_records_every_step():
    result = PipelineExecutor().run([{"id": 1, "value": 1.0}])
    assert result["trace"] == topological_order()
    assert result["registered_model"]["metric"] == "wrmsfe"
    assert result["ingest"]["entity"] == "kpi_series"


def test_cli_plan_command():
    from typer.testing import CliRunner

    runner = CliRunner()
    result = runner.invoke(app, ["plan"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout
