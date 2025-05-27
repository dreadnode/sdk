"""
Tests for logging and integration functionality.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

import dreadnode
from dreadnode.tracing.exporters import FileMetricReader, FileSpanExporter


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = MagicMock()
    with patch("dreadnode.util.logger", logger):
        yield logger


def test_log_param(configured_instance):
    """Test logging parameters."""
    # Create a run to log parameters
    with configured_instance.run("param-test") as run:
        # Log a parameter
        configured_instance.log_param("learning_rate", 0.01)

        # Verify the parameter was logged
        params = run.get_params()
        assert "learning_rate" in params
        assert params["learning_rate"] == 0.01


def test_log_params(configured_instance):
    """Test logging multiple parameters at once."""
    with configured_instance.run("multi-param-test") as run:
        # Log multiple parameters
        configured_instance.log_params({"batch_size": 32, "epochs": 10, "optimizer": "adam"})

        # Verify all parameters were logged
        params = run.get_params()
        assert "batch_size" in params
        assert "epochs" in params
        assert "optimizer" in params
        assert params["batch_size"] == 32
        assert params["epochs"] == 10
        assert params["optimizer"] == "adam"


def test_log_input(configured_instance):
    """Test logging inputs."""
    with configured_instance.run("input-test") as run:
        # Log an input
        configured_instance.log_input("image", {"url": "http://example.com/image.jpg"})

        # Verify the input was logged
        inputs = run.get_inputs()
        assert "image" in inputs
        assert inputs["image"]["url"] == "http://example.com/image.jpg"


def test_log_inputs(configured_instance):
    """Test logging multiple inputs at once."""
    with configured_instance.run("multi-input-test") as run:
        # Log multiple inputs
        configured_instance.log_inputs(
            {
                "image": {"url": "http://example.com/image.jpg"},
                "prompt": "Generate a caption for the image",
            }
        )

        # Verify all inputs were logged
        inputs = run.get_inputs()
        assert "image" in inputs
        assert "prompt" in inputs
        assert inputs["image"]["url"] == "http://example.com/image.jpg"
        assert inputs["prompt"] == "Generate a caption for the image"


def test_log_output(configured_instance):
    """Test logging outputs."""
    with configured_instance.run("output-test") as run:
        # Log an output
        configured_instance.log_output("caption", "A beautiful sunset over mountains")

        # Verify the output was logged
        outputs = run.get_outputs()
        assert "caption" in outputs
        assert outputs["caption"] == "A beautiful sunset over mountains"


def test_file_exporters(tmp_path):
    """Test file exporters for spans and metrics."""
    # Create exporters writing to the temporary directory
    metric_file = tmp_path / "metrics.jsonl"
    span_file = tmp_path / "spans.jsonl"

    # Create exporters
    metric_reader = FileMetricReader(str(metric_file))
    span_exporter = FileSpanExporter(str(span_file))

    # Create an instance using these exporters
    instance = dreadnode.Dreadnode()
    instance.configure(
        project="export-test",
        metric_exporters=[metric_reader],
        span_exporters=[span_exporter],
    )

    # Generate some data
    with instance.run("export-run"):
        instance.log_param("param1", "value1")
        instance.log_metric("metric1", 0.5)

        with instance.task_span("export-task"):
            instance.log_metric("task_metric", 0.75)

    # Shutdown to ensure everything is flushed
    instance.shutdown()

    # Verify the metric file contains the expected data
    assert metric_file.exists()
    metrics = []
    with open(metric_file) as f:
        for line in f:
            metrics.append(json.loads(line))

    assert len(metrics) >= 2  # We should have at least our two metrics

    # Find our metrics
    metric_names = [m.get("name") for m in metrics]
    assert "metric1" in metric_names
    assert "task_metric" in metric_names

    # Verify the span file contains the expected data
    assert span_file.exists()
    spans = []
    with open(span_file) as f:
        for line in f:
            spans.append(json.loads(line))

    assert len(spans) >= 2  # Should have at least run and task spans

    # Find spans by name
    span_names = [s.get("name") for s in spans]
    assert "export-run" in span_names
    assert "export-task" in span_names


def test_push_update(configured_instance):
    """Test the push_update functionality."""
    with configured_instance.run("update-test") as run:
        # Initial state
        configured_instance.log_param("state", "initial")

        # Push an update
        configured_instance.push_update({"state": "updated"})

        # Verify the param was updated
        params = run.get_params()
        assert params["state"] == "updated"

        # Update multiple items
        configured_instance.push_update({"state": "final", "new_param": "value"})

        # Verify all updates were applied
        params = run.get_params()
        assert params["state"] == "final"
        assert params["new_param"] == "value"


def test_integrations_transformers():
    """Test the transformers integration if available."""
    try:
        import transformers

        has_transformers = True
    except ImportError:
        has_transformers = False

    if not has_transformers:
        pytest.skip("Transformers not available")

    from dreadnode.integrations.transformers import DreadnodeCallback

    # Create a mock trainer
    trainer = MagicMock()
    trainer.args = MagicMock()
    trainer.args.output_dir = "/tmp/model"
    trainer.args.logging_steps = 10

    # Create the callback
    callback = DreadnodeCallback()

    # Test on_init_end
    callback.on_init_end(trainer)

    # Test on_log
    callback.on_log(trainer, {"loss": 0.5, "learning_rate": 0.0001})


def test_error_logging(mock_logger, configured_instance):
    """Test error handling and logging."""
    try:
        with configured_instance.run("error-test"):
            # Simulate an error
            raise ValueError("Test error")
    except ValueError:
        pass

    # The logger should have been called with an error message
    mock_logger.error.assert_called()


@pytest.mark.asyncio
async def test_async_logging():
    """Test logging in async contexts."""
    instance = dreadnode.Dreadnode()

    async with instance.run("async-test") as run:
        # Log metrics in async context
        await instance.log_metric_async("async_metric", 0.9)

        # Verify the metric was logged
        metrics = run.get_metrics()
        assert "async_metric" in metrics
        assert metrics.get("async_metric").value == 0.9
