"""Tests for the Task module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from dreadnode.core.task import Task, TaskSpanList, task, TaskFailedWarning
from dreadnode.core.scorer import Scorer
from dreadnode.core.types.common import INHERITED
from dreadnode.core.tracing.span import get_default_tracer


# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def tracer():
    """Get default tracer for tests."""
    return get_default_tracer()


@pytest.fixture
def simple_async_func():
    """A simple async function."""
    async def func(x: int) -> int:
        return x * 2

    return func


@pytest.fixture
def simple_sync_func():
    """A simple sync function."""
    def func(x: int) -> int:
        return x * 2

    return func


@pytest.fixture
def failing_func():
    """A function that always fails."""
    async def func(x: int) -> int:
        raise ValueError("Task failed!")

    return func


@pytest.fixture
def simple_scorer():
    """A simple scorer that returns value / 10."""
    @Scorer
    async def score(value: int) -> float:
        return min(value / 10.0, 1.0)

    return score


@pytest.fixture
def simple_task(tracer, simple_async_func):
    """A simple task for testing."""
    return Task(
        func=simple_async_func,
        tracer=tracer,
        name="simple_task",
    )


# ==============================================================================
# Task Creation Tests
# ==============================================================================


class TestTaskCreation:
    """Tests for Task creation."""

    def test_create_from_async_function(self, tracer, simple_async_func):
        """Test creating task from async function."""
        t = Task(func=simple_async_func, tracer=tracer)

        assert t.name is not None
        assert t.label is not None

    def test_create_from_sync_function(self, tracer, simple_sync_func):
        """Test creating task from sync function."""
        t = Task(func=simple_sync_func, tracer=tracer)

        assert t.name is not None

    def test_create_with_name(self, tracer, simple_async_func):
        """Test creating task with custom name."""
        t = Task(func=simple_async_func, tracer=tracer, name="custom_name")

        assert t.name == "custom_name"

    def test_create_with_label(self, tracer, simple_async_func):
        """Test creating task with custom label."""
        t = Task(func=simple_async_func, tracer=tracer, label="custom_label")

        assert t.label == "custom_label"

    def test_create_with_tags(self, tracer, simple_async_func):
        """Test creating task with tags."""
        t = Task(func=simple_async_func, tracer=tracer, tags=["tag1", "tag2"])

        assert t.tags == ["tag1", "tag2"]

    def test_create_with_attributes(self, tracer, simple_async_func):
        """Test creating task with attributes."""
        t = Task(
            func=simple_async_func,
            tracer=tracer,
            attributes={"key": "value"},
        )

        assert t.attributes["key"] == "value"
        # Should also have code.function attribute
        assert "code.function" in t.attributes

    def test_create_with_scorers(self, tracer, simple_async_func, simple_scorer):
        """Test creating task with scorers."""
        t = Task(func=simple_async_func, tracer=tracer, scorers=[simple_scorer])

        assert len(t.scorers) == 1
        assert t.scorers[0].name == "score"

    def test_create_with_assert_scores(self, tracer, simple_async_func, simple_scorer):
        """Test creating task with assert_scores."""
        t = Task(
            func=simple_async_func,
            tracer=tracer,
            scorers=[simple_scorer],
            assert_scores=["score"],
        )

        assert t.assert_scores == ["score"]

    def test_create_with_assert_scores_true(self, tracer, simple_async_func, simple_scorer):
        """Test creating task with assert_scores=True."""
        t = Task(
            func=simple_async_func,
            tracer=tracer,
            scorers=[simple_scorer],
            assert_scores=True,
        )

        assert t.assert_scores == ["score"]

    def test_create_with_invalid_assert_scores_raises(
        self, tracer, simple_async_func, simple_scorer
    ):
        """Test that invalid assert_scores raises ValueError."""
        with pytest.raises(ValueError, match="Unknown 'invalid'"):
            Task(
                func=simple_async_func,
                tracer=tracer,
                scorers=[simple_scorer],
                assert_scores=["invalid"],
            )

    def test_create_with_log_inputs(self, tracer, simple_async_func):
        """Test creating task with log_inputs."""
        t = Task(func=simple_async_func, tracer=tracer, log_inputs=True)

        assert t.log_inputs is True

    def test_create_with_log_inputs_list(self, tracer, simple_async_func):
        """Test creating task with log_inputs as list."""
        t = Task(func=simple_async_func, tracer=tracer, log_inputs=["x"])

        assert t.log_inputs == ["x"]

    def test_create_with_log_output(self, tracer, simple_async_func):
        """Test creating task with log_output."""
        t = Task(func=simple_async_func, tracer=tracer, log_output=True)

        assert t.log_output is True

    def test_create_with_log_execution_metrics(self, tracer, simple_async_func):
        """Test creating task with log_execution_metrics."""
        t = Task(
            func=simple_async_func,
            tracer=tracer,
            log_execution_metrics=True,
        )

        assert t.log_execution_metrics is True

    def test_create_with_entrypoint(self, tracer, simple_async_func):
        """Test creating task with entrypoint."""
        t = Task(func=simple_async_func, tracer=tracer, entrypoint=True)

        assert t.entrypoint is True

    def test_create_from_generator_raises(self, tracer):
        """Test that creating task from generator raises TypeError."""

        def gen():
            yield 1

        with pytest.raises(TypeError, match="cannot be applied to generators"):
            Task(func=gen, tracer=tracer)

    def test_create_from_async_generator_raises(self, tracer):
        """Test that creating task from async generator raises TypeError."""

        async def gen():
            yield 1

        with pytest.raises(TypeError, match="cannot be applied to generators"):
            Task(func=gen, tracer=tracer)


class TestTaskDecorator:
    """Tests for the @task decorator."""

    def test_decorator_basic(self, tracer):
        """Test basic decorator usage."""

        @task
        async def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)
        assert "my_task" in my_task.name

    def test_decorator_with_options(self, tracer):
        """Test decorator with options."""

        @task(name="custom_name", tags=["test"])
        async def my_task(x: int) -> int:
            return x * 2

        assert my_task.name == "custom_name"
        assert "test" in my_task.tags

    def test_decorator_on_sync_function(self, tracer):
        """Test decorator on sync function."""

        @task
        def my_task(x: int) -> int:
            return x * 2

        assert isinstance(my_task, Task)

    def test_decorator_on_existing_task(self, tracer, simple_task):
        """Test decorator on existing task adds attributes."""

        result = task(
            simple_task,
            tags=["new_tag"],
            name="overridden_name",
        )

        assert isinstance(result, Task)
        assert "new_tag" in result.tags


# ==============================================================================
# Task Properties Tests
# ==============================================================================


class TestTaskProperties:
    """Tests for Task properties."""

    def test_repr(self, simple_task):
        """Test __repr__ method."""
        result = repr(simple_task)

        assert "Task(" in result
        assert "name='simple_task'" in result

    def test_repr_with_scorers(self, tracer, simple_async_func, simple_scorer):
        """Test __repr__ with scorers."""
        t = Task(
            func=simple_async_func,
            tracer=tracer,
            scorers=[simple_scorer],
        )

        result = repr(t)

        assert "scorers=" in result

    def test_repr_with_entrypoint(self, tracer, simple_async_func):
        """Test __repr__ with entrypoint."""
        t = Task(
            func=simple_async_func,
            tracer=tracer,
            entrypoint=True,
        )

        result = repr(t)

        assert "entrypoint=True" in result


# ==============================================================================
# Task Clone and With Tests
# ==============================================================================


class TestTaskCloneAndWith:
    """Tests for Task.clone() and Task.with_() methods."""

    def test_clone(self, simple_task):
        """Test cloning a task."""
        cloned = simple_task.clone()

        assert cloned is not simple_task
        assert cloned.name == simple_task.name
        assert cloned.label == simple_task.label

    def test_with_name(self, simple_task):
        """Test with_ changing name."""
        result = simple_task.with_(name="new_name")

        assert result.name == "new_name"
        assert simple_task.name == "simple_task"  # Original unchanged

    def test_with_label(self, simple_task):
        """Test with_ changing label."""
        result = simple_task.with_(label="new_label")

        assert result.label == "new_label"

    def test_with_tags_replace(self, simple_task):
        """Test with_ replacing tags."""
        result = simple_task.with_(tags=["new_tag"])

        assert result.tags == ["new_tag"]

    def test_with_tags_append(self, tracer, simple_async_func):
        """Test with_ appending tags."""
        t = Task(
            func=simple_async_func,
            tracer=tracer,
            tags=["original"],
        )

        result = t.with_(tags=["new_tag"], append=True)

        assert "original" in result.tags
        assert "new_tag" in result.tags

    def test_with_scorers_replace(self, simple_task, simple_scorer):
        """Test with_ replacing scorers."""
        result = simple_task.with_(scorers=[simple_scorer])

        assert len(result.scorers) == 1

    def test_with_scorers_append(self, tracer, simple_async_func, simple_scorer):
        """Test with_ appending scorers."""

        @Scorer
        async def other_scorer(x: int) -> float:
            return 0.5

        t = Task(
            func=simple_async_func,
            tracer=tracer,
            scorers=[simple_scorer],
        )

        result = t.with_(scorers=[other_scorer], append=True)

        assert len(result.scorers) == 2

    def test_with_log_inputs(self, simple_task):
        """Test with_ changing log_inputs."""
        result = simple_task.with_(log_inputs=True)

        assert result.log_inputs is True

    def test_with_log_output(self, simple_task):
        """Test with_ changing log_output."""
        result = simple_task.with_(log_output=True)

        assert result.log_output is True

    def test_with_attributes_replace(self, simple_task):
        """Test with_ replacing attributes."""
        result = simple_task.with_(attributes={"new": "attr"})

        assert result.attributes == {"new": "attr"}

    def test_with_attributes_append(self, tracer, simple_async_func):
        """Test with_ appending attributes."""
        t = Task(
            func=simple_async_func,
            tracer=tracer,
            attributes={"original": "value"},
        )

        result = t.with_(attributes={"new": "attr"}, append=True)

        # code.function is added automatically
        assert result.attributes["new"] == "attr"


# ==============================================================================
# Task Execution Tests
# ==============================================================================


class TestTaskExecution:
    """Tests for Task execution methods."""

    @pytest.mark.asyncio
    async def test_call(self, simple_task):
        """Test calling task directly."""
        result = await simple_task(5)

        assert result == 10

    @pytest.mark.asyncio
    async def test_run(self, simple_task):
        """Test run method."""
        span = await simple_task.run(5)

        assert span.output == 10
        assert span.exception is None

    @pytest.mark.asyncio
    async def test_run_always_success(self, simple_task):
        """Test run_always with successful execution."""
        span = await simple_task.run_always(5)

        assert span.output == 10
        assert span.exception is None

    @pytest.mark.asyncio
    async def test_run_always_failure(self, tracer, failing_func):
        """Test run_always with failed execution."""
        t = Task(func=failing_func, tracer=tracer)

        span = await t.run_always(5)

        assert span.exception is not None
        assert "Task failed!" in str(span.exception)

    @pytest.mark.asyncio
    async def test_run_raises_on_failure(self, tracer, failing_func):
        """Test run raises exception on failure."""
        t = Task(func=failing_func, tracer=tracer)

        with pytest.raises(ValueError, match="Task failed!"):
            await t.run(5)

    @pytest.mark.asyncio
    async def test_try_(self, simple_task):
        """Test try_ method with successful execution."""
        result = await simple_task.try_(5)

        assert result == 10

    @pytest.mark.asyncio
    async def test_try_failure_returns_none(self, tracer, failing_func):
        """Test try_ returns None on failure."""
        t = Task(func=failing_func, tracer=tracer)

        result = await t.try_(5)

        assert result is None

    @pytest.mark.asyncio
    async def test_sync_function_execution(self, tracer, simple_sync_func):
        """Test executing a sync function wrapped as task."""
        t = Task(func=simple_sync_func, tracer=tracer)

        result = await t(5)

        assert result == 10


# ==============================================================================
# Task Retry Tests
# ==============================================================================


class TestTaskRetry:
    """Tests for Task.retry() method."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_first_attempt(self, simple_task):
        """Test retry succeeds on first attempt."""
        result = await simple_task.retry(3, 5)

        assert result == 10

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_failures(self, tracer):
        """Test retry succeeds after some failures."""
        attempts = 0

        async def flaky_func(x: int) -> int:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ValueError("Temporary failure")
            return x * 2

        t = Task(func=flaky_func, tracer=tracer)

        result = await t.retry(5, 5)

        assert result == 10
        assert attempts == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self, tracer, failing_func):
        """Test retry raises after all attempts exhausted."""
        t = Task(func=failing_func, tracer=tracer)

        with pytest.raises(ValueError, match="Task failed!"):
            await t.retry(3, 5)


# ==============================================================================
# Task Many Tests
# ==============================================================================


class TestTaskMany:
    """Tests for Task.many() and related methods."""

    @pytest.mark.asyncio
    async def test_many(self, simple_task):
        """Test many method."""
        results = await simple_task.many(3, 5)

        assert len(results) == 3
        assert all(r == 10 for r in results)

    @pytest.mark.asyncio
    async def test_try_many(self, tracer):
        """Test try_many excludes failures."""
        attempts = 0

        async def sometimes_fails(x: int) -> int:
            nonlocal attempts
            attempts += 1
            if attempts % 2 == 0:
                raise ValueError("Failure")
            return x * 2

        t = Task(func=sometimes_fails, tracer=tracer)

        results = await t.try_many(4, 5)

        # Only odd attempts succeed
        assert len(results) == 2
        assert all(r == 10 for r in results)


# ==============================================================================
# Task Map Tests
# ==============================================================================


class TestTaskMap:
    """Tests for Task.map() and related methods."""

    @pytest.mark.asyncio
    async def test_map_with_list(self, simple_task):
        """Test map with simple list."""
        results = await simple_task.map([1, 2, 3])

        # Results may be in any order due to concurrency
        assert sorted(results) == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_map_with_dict(self, tracer):
        """Test map with dict of params."""

        async def func(a: int, b: int) -> int:
            return a + b

        t = Task(func=func, tracer=tracer)

        results = await t.map({"a": [1, 2, 3], "b": [10, 20, 30]})

        # Results may be in any order due to concurrency
        assert sorted(results) == [11, 22, 33]

    @pytest.mark.asyncio
    async def test_map_with_static_param(self, tracer):
        """Test map with static parameter."""

        async def func(a: int, b: int) -> int:
            return a + b

        t = Task(func=func, tracer=tracer)

        results = await t.map({"a": [1, 2, 3], "b": 10})

        # Results may be in any order due to concurrency
        assert sorted(results) == [11, 12, 13]

    @pytest.mark.asyncio
    async def test_map_mismatched_lengths_raises(self, tracer):
        """Test map with mismatched list lengths raises."""

        async def func(a: int, b: int) -> int:
            return a + b

        t = Task(func=func, tracer=tracer)

        with pytest.raises(ValueError, match="Mismatched lengths"):
            await t.map({"a": [1, 2, 3], "b": [1, 2]})

    @pytest.mark.asyncio
    async def test_map_no_list_raises(self, tracer):
        """Test map with no list raises."""

        async def func(a: int) -> int:
            return a

        t = Task(func=func, tracer=tracer)

        with pytest.raises(ValueError, match="at least one list"):
            await t.map({"a": 1})

    @pytest.mark.asyncio
    async def test_map_invalid_type_raises(self, tracer):
        """Test map with invalid type raises."""

        async def func(a: int) -> int:
            return a

        t = Task(func=func, tracer=tracer)

        with pytest.raises(TypeError, match="Expected 'args' to be a list or dict"):
            await t.map("invalid")  # type: ignore

    @pytest.mark.asyncio
    async def test_try_map(self, tracer):
        """Test try_map excludes failures."""
        call_count = 0

        async def sometimes_fails(x: int) -> int:
            nonlocal call_count
            call_count += 1
            if x == 2:
                raise ValueError("Failure")
            return x * 2

        t = Task(func=sometimes_fails, tracer=tracer)

        results = await t.try_map([1, 2, 3])

        # Only 1 and 3 succeed
        assert len(results) == 2
        assert 2 in results  # 1 * 2
        assert 6 in results  # 3 * 2

    @pytest.mark.asyncio
    async def test_map_with_concurrency(self, tracer):
        """Test map with concurrency limit."""

        async def func(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        t = Task(func=func, tracer=tracer)

        results = await t.map([1, 2, 3, 4], concurrency=2)

        # Results may be in any order due to concurrency
        assert sorted(results) == [2, 4, 6, 8]


# ==============================================================================
# TaskSpanList Tests
# ==============================================================================


class TestTaskSpanList:
    """Tests for TaskSpanList class."""

    def test_is_list(self):
        """Test TaskSpanList inherits from list."""
        spans = TaskSpanList()

        assert isinstance(spans, list)
        assert isinstance(spans, TaskSpanList)

    def test_append_and_extend(self):
        """Test list operations work."""
        spans = TaskSpanList()

        # Just testing it works as a list - actual span sorting
        # requires metrics which need full tracing setup
        spans.append("item1")
        spans.extend(["item2", "item3"])

        assert len(spans) == 3


# ==============================================================================
# Task as_eval Tests
# ==============================================================================


class TestTaskAsEval:
    """Tests for Task.as_eval() method."""

    def test_as_eval_basic(self, tracer, simple_async_func):
        """Test as_eval creates Evaluation."""
        from dreadnode.core.evaluations.evaluation import Evaluation

        t = Task(func=simple_async_func, tracer=tracer)
        eval_ = t.as_eval(name="test_eval", dataset=[1, 2, 3])

        assert isinstance(eval_, Evaluation)
        assert eval_.task is not None

    def test_as_eval_with_options(self, tracer, simple_async_func):
        """Test as_eval with options."""
        t = Task(func=simple_async_func, tracer=tracer)
        eval_ = t.as_eval(
            dataset=[1, 2, 3],
            name="test_eval",
            description="Test description",
            concurrency=5,
            iterations=3,
        )

        assert eval_.name == "test_eval"
        assert eval_.description == "Test description"
        assert eval_.concurrency == 5
        assert eval_.iterations == 3


# ==============================================================================
# Task Descriptor Tests
# ==============================================================================


class TestTaskDescriptor:
    """Tests for Task as class descriptor."""

    def test_get_on_class(self, tracer):
        """Test __get__ on class returns Task."""

        class MyClass:
            @task
            async def my_method(self, x: int) -> int:
                return x * 2

        assert isinstance(MyClass.my_method, Task)

    @pytest.mark.asyncio
    async def test_get_on_instance(self, tracer):
        """Test __get__ on instance returns bound Task."""

        class MyClass:
            @task
            async def my_method(self, x: int) -> int:
                return x * 2

        obj = MyClass()

        assert isinstance(obj.my_method, Task)

        # Should work when called
        result = await obj.my_method(5)
        assert result == 10


# ==============================================================================
# Task deepcopy Tests
# ==============================================================================


class TestTaskDeepCopy:
    """Tests for Task deep copying."""

    def test_deepcopy(self, simple_task):
        """Test __deepcopy__ method."""
        import copy

        copied = copy.deepcopy(simple_task)

        assert copied is not simple_task
        assert copied.name == simple_task.name
        assert copied.tags is not simple_task.tags
        assert copied.attributes is not simple_task.attributes
