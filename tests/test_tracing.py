"""
Tests for the tracing module and span functionality.
"""

from datetime import datetime

import pytest

import dreadnode
from dreadnode.tracing.span import RunSpan, Span, TaskSpan


def test_span_creation() -> None:
    """Test creating a span."""
    span = Span(
        name="test-span",
        start_time=datetime.now().timestamp(),
        span_id="test-span-id",
        trace_id="test-trace-id",
    )

    assert span.name == "test-span"
    assert span.span_id == "test-span-id"
    assert span.trace_id == "test-trace-id"
    assert span.end_time is None  # Not ended yet
    assert span.is_error is False


def test_span_context_manager(configured_instance) -> None:
    """Test using span as a context manager."""
    with configured_instance.span("test-operation") as span:
        assert span.name == "test-operation"
        assert span.end_time is None  # Not ended within the context

    # After exiting context, span should be ended
    assert span.end_time is not None
    assert span.is_error is False


def test_span_error_handling(configured_instance):
    """Test error handling in span."""
    try:
        with configured_instance.span("error-operation") as span:
            raise ValueError("Test error")
    except ValueError:
        pass

    # Span should be ended and marked as error
    assert span.end_time is not None
    assert span.is_error is True
    assert "error" in span.attributes
    assert "ValueError: Test error" in span.attributes["error"]


def test_run_span(configured_instance):
    """Test RunSpan functionality."""
    with configured_instance.run(name="test-run", tags={"env": "test"}) as run:
        assert isinstance(run, RunSpan)
        assert run.name == "test-run"
        assert run.attributes["env"] == "test"

        # Log metrics within run
        configured_instance.log_metric("accuracy", 0.9)

    # After run completion
    assert run.end_time is not None
    metrics = run.get_metrics()
    assert "accuracy" in metrics
    assert metrics.get("accuracy").value == 0.9


def test_task_span(configured_instance):
    """Test TaskSpan functionality."""
    with configured_instance.task_span("test-task") as task_span:
        assert isinstance(task_span, TaskSpan)
        assert task_span.name == "test-task"

        # Log metrics and params
        configured_instance.log_metric("precision", 0.8)
        configured_instance.log_param("batch_size", 32)

    # After task completion
    assert task_span.end_time is not None
    metrics = task_span.get_metrics()
    assert "precision" in metrics
    assert metrics.get("precision").value == 0.8

    params = task_span.get_params()
    assert "batch_size" in params
    assert params["batch_size"] == 32


def test_nested_spans(configured_instance):
    """Test nesting spans within each other."""
    with configured_instance.run(name="parent-run") as parent:
        # Log parent metric
        configured_instance.log_metric("parent-metric", 1.0)

        with configured_instance.task_span("child-task") as child:
            # Log child metric
            configured_instance.log_metric("child-metric", 0.5)

        # Check child metrics
        child_metrics = child.get_metrics()
        assert "child-metric" in child_metrics
        assert child_metrics.get("child-metric").value == 0.5

    # Check parent metrics
    parent_metrics = parent.get_metrics()
    assert "parent-metric" in parent_metrics
    assert parent_metrics.get("parent-metric").value == 1.0


def test_current_spans(configured_instance):
    """Test accessing current spans."""
    with configured_instance.run(name="test-run"):
        run_span = dreadnode.tracing.span.current_run_span()
        assert run_span is not None
        assert run_span.name == "test-run"

        with configured_instance.task_span("test-task"):
            task_span = dreadnode.tracing.span.current_task_span()
            assert task_span is not None
            assert task_span.name == "test-task"


def test_run_multiple_tasks(configured_instance):
    """Test running multiple tasks within a run."""
    with configured_instance.run(name="multi-task-run") as run:
        # Define task functions
        @configured_instance.task
        def task1(x):
            return x * 2

        @configured_instance.task
        def task2(x):
            return x + 10

        # Execute tasks
        result1 = task1(5)
        result2 = task2(5)

        assert result1 == 10
        assert result2 == 15

    # Check that task spans are accessible
    assert len(task1.spans) == 1
    assert len(task2.spans) == 1

    # Check that task results match
    assert task1.spans[0].result == 10
    assert task2.spans[0].result == 15


@pytest.mark.asyncio
async def test_async_span():
    """Test async spans."""
    instance = dreadnode.Dreadnode()

    async with instance.span("async-span") as span:
        assert span.name == "async-span"
        assert span.end_time is None

    # After exiting async context, span should be ended
    assert span.end_time is not None
