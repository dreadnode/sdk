"""
Tests for Task output object hash linking logic.

This test module covers the fix for ENG-3549, which addresses a scope issue
where output_object_hash could be referenced before definition.

The bug: output_object_hash was only defined inside the `if log_output and ...`
block, but was referenced in the subsequent `elif` block, causing UnboundLocalError.

The fix: Initialize output_object_hash = None before use.
"""

import pytest

from dreadnode import task


@pytest.mark.asyncio
async def test_task_with_log_output_true() -> None:
    """Test that a task with log_output=True executes without errors."""

    @task(log_inputs=True, log_output=True)
    def sample_task(x: int) -> int:
        return x * 2

    result = await sample_task.run_always(5)
    assert result.output == 10


@pytest.mark.asyncio
async def test_task_with_log_output_false() -> None:
    """Edge case where output_object_hash would not be defined in buggy code."""

    @task(log_inputs=True, log_output=False)
    def sample_task(x: int) -> int:
        return x * 2

    result = await sample_task.run_always(5)
    assert result.output == 10


@pytest.mark.asyncio
async def test_task_with_no_logging() -> None:
    """
    Core bug scenario: no logging means output_object_hash would be
    referenced before definition in the original buggy code.
    """

    @task(log_inputs=False, log_output=False)
    def sample_task(x: int) -> int:
        return x * 2

    result = await sample_task.run_always(5)
    assert result.output == 10


@pytest.mark.asyncio
async def test_task_with_multiple_inputs() -> None:
    """Test that linking logic handles multiple input hashes properly."""

    @task(log_inputs=True, log_output=True)
    def sample_task(x: int, y: int, z: int) -> int:
        return x + y + z

    result = await sample_task.run_always(1, 2, 3)
    assert result.output == 6


@pytest.mark.asyncio
async def test_async_task_execution() -> None:
    """Test that the fix works correctly for async tasks."""

    @task(log_inputs=True, log_output=True)
    async def async_sample_task(x: int) -> int:
        return x * 2

    result = await async_sample_task.run_always(5)
    assert result.output == 10


@pytest.mark.asyncio
async def test_task_with_inherited_log_settings() -> None:
    """Test inherited logging settings (the default, most common usage)."""

    @task
    def sample_task(x: int) -> int:
        return x * 2

    result = await sample_task.run_always(5)
    assert result.output == 10


@pytest.mark.asyncio
async def test_task_exception_handling() -> None:
    """Test that exceptions don't cause issues with output_object_hash logic."""

    @task(log_inputs=True, log_output=True)
    def failing_task(_x: int) -> int:
        raise ValueError("Intentional test error")

    result = await failing_task.run_always(5)

    assert result.exception is not None
    assert isinstance(result.exception, ValueError)
    assert "Intentional test error" in str(result.exception)


@pytest.mark.asyncio
async def test_task_with_complex_output() -> None:
    """Test that tasks returning complex types work correctly."""

    @task(log_inputs=True, log_output=True)
    def complex_task(x: int) -> dict[str, int]:
        return {"result": x * 2, "input": x}

    result = await complex_task.run_always(5)
    assert result.output == {"result": 10, "input": 5}


@pytest.mark.asyncio
async def test_task_with_none_output() -> None:
    """Test None outputs (may be handled differently in serialization)."""

    @task(log_inputs=True, log_output=True)
    def none_task(x: int) -> None:
        pass

    result = await none_task.run_always(5)
    assert result.output is None


@pytest.mark.asyncio
async def test_task_entrypoint_behavior() -> None:
    """Test entrypoint tasks (create_run=True path) with log_output=False."""

    @task(log_inputs=True, log_output=False, entrypoint=True)
    def entrypoint_task(x: int) -> int:
        return x * 2

    result = await entrypoint_task.run_always(5)
    assert result.output == 10
