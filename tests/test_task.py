"""
Tests for the Task module and related functionality.
"""

import pytest

import dreadnode
from dreadnode.task import Task, TaskSpanList
from dreadnode.tracing.span import TaskSpan


def test_task_creation() -> None:
    """Test creating a task with various configurations."""

    # Test creating a task with basic function
    def add(a, b):
        return a + b

    task = Task(add)
    assert task.name == "add"
    assert task.func is add

    # Test with custom name
    task = Task(add, name="custom_add")
    assert task.name == "custom_add"
    assert task.func is add

    # Test task execution
    result = task(1, 2)
    assert result == 3


def test_task_decorator() -> None:
    """Test using Task as a decorator."""

    @Task
    def multiply(a, b):
        return a * b

    assert isinstance(multiply, Task)
    assert multiply.name == "multiply"
    assert multiply(2, 3) == 6

    # Test with custom name
    @Task(name="custom_multiply")
    def multiply2(a, b):
        return a * b

    assert multiply2.name == "custom_multiply"
    assert multiply2(2, 3) == 6


def test_task_with_instance(configured_instance) -> None:
    """Test task execution with dreadnode instance."""

    @configured_instance.task
    def divide(a, b):
        return a / b

    # Execute the task
    result = divide(10, 2)
    assert result == 5

    # Check that spans are recorded
    assert divide.spans is not None
    assert len(divide.spans) > 0
    assert isinstance(divide.spans[0], TaskSpan)
    assert divide.spans[0].result == 5


def test_task_with_metrics(configured_instance) -> None:
    """Test task with metrics recording."""

    @configured_instance.task
    def predict(input_val):
        configured_instance.log_metric("accuracy", 0.8)
        configured_instance.log_metric("loss", 0.2)
        return input_val * 2

    result = predict(5)
    assert result == 10

    # Check that metrics were recorded in the span
    span = predict.spans[0]
    metrics = span.get_metrics()
    assert "accuracy" in metrics
    assert "loss" in metrics
    assert metrics.get("accuracy").value == 0.8
    assert metrics.get("loss").value == 0.2


def test_task_with_params(configured_instance) -> None:
    """Test task with parameter logging."""

    @configured_instance.task
    def train_model(learning_rate, epochs):
        configured_instance.log_param("learning_rate", learning_rate)
        configured_instance.log_param("epochs", epochs)
        return learning_rate * epochs

    result = train_model(0.01, 10)
    assert result == 0.1

    # Check that parameters were recorded
    span = train_model.spans[0]
    params = span.get_params()
    assert "learning_rate" in params
    assert "epochs" in params
    assert params["learning_rate"] == 0.01
    assert params["epochs"] == 10


def test_multiple_task_executions(configured_instance):
    """Test executing a task multiple times."""

    @configured_instance.task
    def score(value):
        return value * 2

    # Execute the task multiple times with different inputs
    results = []
    for i in range(5):
        results.append(score(i))

    assert results == [0, 2, 4, 6, 8]
    assert len(score.spans) == 5

    # Test TaskSpanList sorting
    spans = TaskSpanList(score.spans)

    # Add metrics to spans with different values
    for i, span in enumerate(spans):
        configured_instance.log_metric(f"score_{i}", i / 10, span=span)

    # Test sorting (default is descending)
    sorted_spans = spans.sorted()
    assert len(sorted_spans) == 5

    # Test top_n
    top_spans = spans.top_n(2)
    assert len(top_spans) == 2

    # Test bottom_n
    bottom_spans = spans.bottom_n(2)
    assert len(bottom_spans) == 2


def test_async_task():
    """Test async task execution."""
    instance = dreadnode.Dreadnode()

    @instance.task
    async def async_task(a, b):
        return a + b

    @pytest.mark.asyncio
    async def test_async_execution():
        result = await async_task(3, 4)
        assert result == 7
        assert len(async_task.spans) == 1

    pytest.main(["-xvs", __file__])


def test_task_failure(configured_instance):
    """Test task failure handling."""

    @configured_instance.task
    def failing_task():
        raise ValueError("Task failed")

    # Execute and expect an exception
    with pytest.raises(ValueError, match="Task failed"):
        failing_task()

    # Check that the span still exists and has error info
    assert len(failing_task.spans) == 1
    span = failing_task.spans[0]
    assert span.is_error is True
    assert "error" in span.attributes
    assert "ValueError: Task failed" in span.attributes["error"]
