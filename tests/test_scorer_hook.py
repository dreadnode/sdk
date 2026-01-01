"""Tests for ScorerHook and related functionality."""

import pytest

from dreadnode.core.agents.scorer_hook import (
    ScorerHook,
    ScorerHookResult,
    ScoreCondition,
    CompositeCondition,
    scorer_on,
)
from dreadnode.core.agents.events import GenerationStep, ToolStep, AgentStep
from dreadnode.core.agents.reactions import Fail, Finish, RetryWithFeedback
from dreadnode.core.generators.message import Message
from dreadnode.core.generators.generator import Usage
from dreadnode.core.metric import Metric
from dreadnode.core.scorer import Scorer


# Fixtures


@pytest.fixture
def simple_scorer():
    """A simple scorer that returns length / 100."""

    @Scorer
    async def length_scorer(text: str) -> float:
        return len(text) / 100.0

    return length_scorer


@pytest.fixture
def always_high_scorer():
    """A scorer that always returns 0.9."""

    @Scorer
    async def high_scorer(obj) -> float:
        return 0.9

    return high_scorer


@pytest.fixture
def always_low_scorer():
    """A scorer that always returns 0.1."""

    @Scorer
    async def low_scorer(obj) -> float:
        return 0.1

    return low_scorer


@pytest.fixture
def generation_step():
    """A sample GenerationStep event."""
    return GenerationStep(
        step=1,
        messages=[Message(role="assistant", content="Hello, world!")],
        usage=Usage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


@pytest.fixture
def tool_step():
    """A sample ToolStep event."""
    from dreadnode.core.tools import ToolCall, FunctionCall

    return ToolStep(
        step=2,
        messages=[Message(role="tool", content="Tool result")],
        tool_call=ToolCall(
            id="call_123",
            function=FunctionCall(name="get_weather", arguments='{"city": "NYC"}'),
        ),
    )


# ScorerHookResult tests


class TestScorerHookResult:
    """Tests for ScorerHookResult dataclass."""

    def test_basic_creation(self):
        """Test creating a basic result."""
        result = ScorerHookResult(scorer_name="quality")

        assert result.scorer_name == "quality"
        assert result.metric is None
        assert result.reaction is None
        assert result.step is None

    def test_with_all_fields(self):
        """Test creating a result with all fields."""
        metric = Metric(value=0.85, step=1)
        reaction = Finish(reason="Score exceeded threshold")

        result = ScorerHookResult(
            scorer_name="quality",
            metric=metric,
            reaction=reaction,
            step=1,
        )

        assert result.scorer_name == "quality"
        assert result.metric.value == 0.85
        assert isinstance(result.reaction, Finish)
        assert result.step == 1


# ScorerHook tests


class TestScorerHook:
    """Tests for ScorerHook class."""

    def test_creation_from_scorer(self, simple_scorer):
        """Test creating a ScorerHook from a Scorer."""
        hook = ScorerHook(
            scorer=simple_scorer,
            event_type=GenerationStep,
        )

        assert hook.scorer.name == "length_scorer"
        assert hook.event_type == GenerationStep
        assert hook.__name__ == "scorer_hook:length_scorer"

    def test_creation_with_adapter(self, simple_scorer):
        """Test creating a ScorerHook with an adapter."""
        hook = ScorerHook(
            scorer=simple_scorer,
            event_type=GenerationStep,
            adapter=lambda e: e.messages[0].content if e.messages else "",
        )

        assert hook.adapter is not None

    @pytest.mark.asyncio
    async def test_call_matching_event(self, always_high_scorer, generation_step):
        """Test calling hook with matching event type."""
        hook = ScorerHook(
            scorer=always_high_scorer,
            event_type=GenerationStep,
            create_span=False,
            log_metrics=False,
        )

        result = await hook(generation_step)

        assert result is not None
        assert result.scorer_name == "high_scorer"
        assert result.metric is not None
        assert result.metric.value == 0.9
        assert result.step == 1

    @pytest.mark.asyncio
    async def test_call_non_matching_event(self, always_high_scorer, tool_step):
        """Test calling hook with non-matching event type returns None."""
        hook = ScorerHook(
            scorer=always_high_scorer,
            event_type=GenerationStep,  # Expecting GenerationStep
            create_span=False,
            log_metrics=False,
        )

        result = await hook(tool_step)  # Passing ToolStep

        assert result is None

    @pytest.mark.asyncio
    async def test_call_with_adapter(self, simple_scorer, generation_step):
        """Test calling hook with adapter extracts correct object."""
        hook = ScorerHook(
            scorer=simple_scorer,
            event_type=GenerationStep,
            adapter=lambda e: e.messages[0].content if e.messages else "",
            create_span=False,
            log_metrics=False,
        )

        result = await hook(generation_step)

        assert result is not None
        # "Hello, world!" has 13 chars, so score should be 0.13
        assert result.metric.value == pytest.approx(0.13, rel=0.01)

    @pytest.mark.asyncio
    async def test_low_threshold_reaction(self, always_low_scorer, generation_step):
        """Test that low threshold triggers reaction."""
        hook = ScorerHook(
            scorer=always_low_scorer,
            event_type=GenerationStep,
            low_threshold=0.5,
            on_low=Fail(error="Score too low"),
            create_span=False,
            log_metrics=False,
        )

        result = await hook(generation_step)

        assert result is not None
        assert isinstance(result.reaction, Fail)
        assert result.reaction.error == "Score too low"

    @pytest.mark.asyncio
    async def test_high_threshold_reaction(self, always_high_scorer, generation_step):
        """Test that high threshold triggers reaction."""
        hook = ScorerHook(
            scorer=always_high_scorer,
            event_type=GenerationStep,
            high_threshold=0.8,
            on_high=Finish(reason="Score exceeded threshold"),
            create_span=False,
            log_metrics=False,
        )

        result = await hook(generation_step)

        assert result is not None
        assert isinstance(result.reaction, Finish)
        assert result.reaction.reason == "Score exceeded threshold"

    @pytest.mark.asyncio
    async def test_callable_reaction(self, always_low_scorer, generation_step):
        """Test using a callable for reaction."""
        hook = ScorerHook(
            scorer=always_low_scorer,
            event_type=GenerationStep,
            low_threshold=0.5,
            on_low=lambda m: RetryWithFeedback(f"Score was {m.value:.2f}"),
            create_span=False,
            log_metrics=False,
        )

        result = await hook(generation_step)

        assert result is not None
        assert isinstance(result.reaction, RetryWithFeedback)
        assert "0.10" in result.reaction.feedback


class TestScorerHookConvenienceMethods:
    """Tests for ScorerHook convenience methods."""

    def test_fail_if_below(self, always_low_scorer):
        """Test fail_if_below convenience method."""
        hook = ScorerHook(
            scorer=always_low_scorer,
            event_type=GenerationStep,
        ).fail_if_below(0.5, error="Too low!")

        assert hook.low_threshold == 0.5
        assert isinstance(hook.on_low, Fail)
        assert hook.on_low.error == "Too low!"

    def test_finish_if_above(self, always_high_scorer):
        """Test finish_if_above convenience method."""
        hook = ScorerHook(
            scorer=always_high_scorer,
            event_type=GenerationStep,
        ).finish_if_above(0.8, reason="Done!")

        assert hook.high_threshold == 0.8
        assert isinstance(hook.on_high, Finish)
        assert hook.on_high.reason == "Done!"

    def test_retry_if_below(self, always_low_scorer):
        """Test retry_if_below convenience method."""
        hook = ScorerHook(
            scorer=always_low_scorer,
            event_type=GenerationStep,
        ).retry_if_below(0.5)

        assert hook.low_threshold == 0.5
        assert callable(hook.on_low)


class TestScorerOn:
    """Tests for Scorer.on() method."""

    def test_scorer_on_creates_hook(self, simple_scorer):
        """Test that Scorer.on() creates a ScorerHook."""
        hook = simple_scorer.on(GenerationStep)

        assert isinstance(hook, ScorerHook)
        assert hook.event_type == GenerationStep
        assert hook.scorer.name == "length_scorer"

    def test_scorer_on_with_adapter(self, simple_scorer):
        """Test Scorer.on() with adapter."""
        adapter = lambda e: e.messages[0].content if e.messages else ""
        hook = simple_scorer.on(GenerationStep, adapter=adapter)

        assert hook.adapter is adapter

    def test_scorer_on_chaining(self, simple_scorer):
        """Test chaining methods on hook returned by Scorer.on()."""
        hook = (
            simple_scorer.on(GenerationStep)
            .fail_if_below(0.3)
            .finish_if_above(0.9)
        )

        assert hook.low_threshold == 0.3
        assert hook.high_threshold == 0.9


class TestScorerOnFunction:
    """Tests for scorer_on() function."""

    def test_scorer_on_function(self, simple_scorer):
        """Test scorer_on() creates a ScorerHook."""
        hook = scorer_on(simple_scorer, GenerationStep)

        assert isinstance(hook, ScorerHook)
        assert hook.event_type == GenerationStep


# ScoreCondition tests


class TestScoreCondition:
    """Tests for ScoreCondition class."""

    @pytest.mark.asyncio
    async def test_gt_condition_passes(self, always_high_scorer, generation_step):
        """Test greater-than condition passes."""
        condition = ScoreCondition(
            scorer=always_high_scorer,
            gt=0.5,
        )

        passed, metric = await condition.evaluate(generation_step)

        assert passed is True
        assert metric.value == 0.9

    @pytest.mark.asyncio
    async def test_gt_condition_fails(self, always_low_scorer, generation_step):
        """Test greater-than condition fails."""
        condition = ScoreCondition(
            scorer=always_low_scorer,
            gt=0.5,
        )

        passed, metric = await condition.evaluate(generation_step)

        assert passed is False
        assert metric.value == 0.1

    @pytest.mark.asyncio
    async def test_lt_condition(self, always_low_scorer, generation_step):
        """Test less-than condition."""
        condition = ScoreCondition(
            scorer=always_low_scorer,
            lt=0.5,
        )

        passed, metric = await condition.evaluate(generation_step)

        assert passed is True

    @pytest.mark.asyncio
    async def test_no_threshold_always_passes(self, always_high_scorer, generation_step):
        """Test that no threshold means always pass."""
        condition = ScoreCondition(scorer=always_high_scorer)

        passed, metric = await condition.evaluate(generation_step)

        assert passed is True

    @pytest.mark.asyncio
    async def test_condition_with_adapter(self, simple_scorer, generation_step):
        """Test condition with adapter."""
        condition = ScoreCondition(
            scorer=simple_scorer,
            gt=0.1,
            adapter=lambda e: e.messages[0].content if e.messages else "",
        )

        passed, metric = await condition.evaluate(generation_step)

        assert passed is True
        assert metric.value == pytest.approx(0.13, rel=0.01)


class TestCompositeCondition:
    """Tests for CompositeCondition class."""

    @pytest.mark.asyncio
    async def test_and_both_pass(self, always_high_scorer, generation_step):
        """Test AND condition when both pass."""
        cond1 = ScoreCondition(scorer=always_high_scorer, gt=0.5)
        cond2 = ScoreCondition(scorer=always_high_scorer, gt=0.8)

        composite = cond1 & cond2

        passed, metrics = await composite.evaluate(generation_step)

        assert passed is True
        assert len(metrics) == 2

    @pytest.mark.asyncio
    async def test_and_one_fails(self, always_high_scorer, always_low_scorer, generation_step):
        """Test AND condition when one fails."""
        cond1 = ScoreCondition(scorer=always_high_scorer, gt=0.5)
        cond2 = ScoreCondition(scorer=always_low_scorer, gt=0.5)

        composite = cond1 & cond2

        passed, metrics = await composite.evaluate(generation_step)

        assert passed is False

    @pytest.mark.asyncio
    async def test_or_one_passes(self, always_high_scorer, always_low_scorer, generation_step):
        """Test OR condition when one passes."""
        cond1 = ScoreCondition(scorer=always_high_scorer, gt=0.5)
        cond2 = ScoreCondition(scorer=always_low_scorer, gt=0.5)

        composite = cond1 | cond2

        passed, metrics = await composite.evaluate(generation_step)

        assert passed is True

    @pytest.mark.asyncio
    async def test_or_both_fail(self, always_low_scorer, generation_step):
        """Test OR condition when both fail."""
        cond1 = ScoreCondition(scorer=always_low_scorer, gt=0.5)
        cond2 = ScoreCondition(scorer=always_low_scorer, gt=0.8)

        composite = cond1 | cond2

        passed, metrics = await composite.evaluate(generation_step)

        assert passed is False

    @pytest.mark.asyncio
    async def test_nested_conditions(self, always_high_scorer, always_low_scorer, generation_step):
        """Test nested composite conditions."""
        cond1 = ScoreCondition(scorer=always_high_scorer, gt=0.5)
        cond2 = ScoreCondition(scorer=always_high_scorer, gt=0.8)
        cond3 = ScoreCondition(scorer=always_low_scorer, lt=0.5)

        # (high > 0.5 AND high > 0.8) OR low < 0.5
        composite = (cond1 & cond2) | cond3

        passed, metrics = await composite.evaluate(generation_step)

        assert passed is True
