"""
Tests for decorators and error handling in the task module.
"""

import asyncio
import inspect
import time
from functools import wraps

import pytest

import dreadnode
from dreadnode.task import Task, TaskFailedWarning, TaskGeneratorWarning, TaskSpanList


def test_task_decorator_attributes():
    """Test that the task decorator properly preserves function attributes."""

    def sample_function(a, b):
        """Sample function docstring."""
        return a + b

    # Apply decorator
    decorated = Task(sample_function)

    # Check that function metadata is preserved
    assert decorated.__name__ == "sample_function"
    assert decorated.__doc__ == "Sample function docstring."
    assert inspect.signature(decorated) == inspect.signature(sample_function)


def test_nested_task_decorators(configured_instance):
    """Test nesting task decorators."""

    @configured_instance.task(name="outer_task")
    def outer():
        @configured_instance.task(name="inner_task")
        def inner(x):
            return x * 2

        result = inner(5)
        return result * 3

    # Execute the outer task
    result = outer()

    # Check results
    assert result == 30  # (5 * 2) * 3

    # Check that both tasks have spans
    assert len(outer.spans) == 1
    assert outer.spans[0].name == "outer_task"

    # Inner task should have been called once
    inner_task = None
    for task in configured_instance._tasks:
        if task.name == "inner_task":
            inner_task = task
            break

    assert inner_task is not None
    assert len(inner_task.spans) == 1
    assert inner_task.spans[0].name == "inner_task"


def test_task_with_custom_decorator(configured_instance):
    """Test that task works correctly with other decorators."""

    # Define a custom decorator
    def my_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Add a prefix to the result
            result = func(*args, **kwargs)
            return f"decorated_{result}"

        return wrapper

    # Apply multiple decorators
    @my_decorator
    @configured_instance.task(name="decorated_task")
    def process_text(text):
        return text.upper()

    result = process_text("hello")

    # Check that both decorators were applied
    assert result == "decorated_HELLO"

    # Check that the task span was recorded
    assert len(process_text.spans) == 1
    assert process_text.spans[0].name == "decorated_task"


def test_task_timing(configured_instance):
    """Test that task execution timing is captured correctly."""

    @configured_instance.task
    def slow_function():
        time.sleep(0.1)  # Short sleep to ensure measurable duration
        return "done"

    # Execute the task
    slow_function()

    # Check the span timing
    span = slow_function.spans[0]
    assert span.end_time is not None
    assert span.duration is not None
    assert span.duration >= 0.1  # Duration should be at least the sleep time


def test_task_exception_handling(configured_instance):
    """Test that exceptions in tasks are properly handled."""

    @configured_instance.task
    def failing_function():
        raise ValueError("Task failed deliberately")

    # Execute the task and expect an exception
    with pytest.raises(ValueError, match="Task failed deliberately"):
        failing_function()

    # Verify the task span was recorded with error info
    assert len(failing_function.spans) == 1
    span = failing_function.spans[0]
    assert span.is_error is True
    assert "error" in span.attributes
    assert "ValueError: Task failed deliberately" in span.attributes["error"]


def test_task_warning_handling(configured_instance):
    """Test handling of task warnings."""
    # Test with a generator function (should trigger warning)
    with pytest.warns(TaskGeneratorWarning):

        @configured_instance.task
        def generator_function():
            yield 1
            yield 2

    # Test handling of TaskFailedWarning
    @configured_instance.task
    def warning_function():
        # Simulate a condition that would trigger a warning
        import warnings

        warnings.warn("Task warning", TaskFailedWarning)
        return "completed with warning"

    with pytest.warns(TaskFailedWarning):
        result = warning_function()

    assert result == "completed with warning"
    assert len(warning_function.spans) == 1


def test_task_input_output_capture(configured_instance):
    """Test that task inputs and outputs are properly captured."""

    @configured_instance.task
    def process_data(input_data, multiplier=2):
        configured_instance.log_input("input_data", input_data)
        result = input_data * multiplier
        configured_instance.log_output("result", result)
        return result

    # Execute the task
    result = process_data(5)

    # Check the result
    assert result == 10

    # Check that inputs and outputs were logged
    span = process_data.spans[0]
    inputs = span.get_inputs()
    outputs = span.get_outputs()

    assert "input_data" in inputs
    assert inputs["input_data"] == 5
    assert "result" in outputs
    assert outputs["result"] == 10


@pytest.mark.asyncio
async def test_async_task_execution():
    """Test execution of async tasks."""
    instance = dreadnode.Dreadnode()

    @instance.task
    async def async_task(delay, value):
        await asyncio.sleep(delay)
        return value * 2

    # Execute the task
    result = await async_task(0.1, 5)

    # Check the result
    assert result == 10

    # Check that the span was recorded
    assert len(async_task.spans) == 1
    span = async_task.spans[0]
    assert span.end_time is not None
    assert span.duration >= 0.1


def test_task_span_list_methods(configured_instance):
    """Test TaskSpanList methods."""

    @configured_instance.task
    def sample_task(value):
        configured_instance.log_metric("score", value)
        return value

    # Execute the task multiple times with different inputs
    values = [0.1, 0.5, 0.3, 0.8, 0.2]
    for val in values:
        sample_task(val)

    # Create a TaskSpanList from the spans
    span_list = TaskSpanList(sample_task.spans)

    # Test sorting (default descending)
    sorted_list = span_list.sorted()
    metrics = [span.get_average_metric_value() for span in sorted_list]
    assert metrics == sorted(metrics, reverse=True)

    # Test sorting ascending
    asc_sorted = span_list.sorted(reverse=False)
    asc_metrics = [span.get_average_metric_value() for span in asc_sorted]
    assert asc_metrics == sorted(metrics)

    # Test top_n
    top_2 = span_list.top_n(2)
    assert len(top_2) == 2
    assert top_2[0].get_average_metric_value() >= top_2[1].get_average_metric_value()

    # Test bottom_n
    bottom_2 = span_list.bottom_n(2)
    assert len(bottom_2) == 2
    assert bottom_2[0].get_average_metric_value() <= bottom_2[1].get_average_metric_value()

    # Test get
    best = span_list.get("best")
    assert best.get_average_metric_value() == max(metrics)

    worst = span_list.get("worst")
    assert worst.get_average_metric_value() == min(metrics)
