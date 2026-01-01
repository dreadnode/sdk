"""Tests for explicit span factories and tracing enhancements."""

import pytest

from dreadnode.core.tracing.spans import (
    generation_span,
    scorer_span,
    tool_span,
    trial_span,
)
from dreadnode.core.tracing.constants import (
    GENERATION_ATTRIBUTE_MODEL,
    GENERATION_ATTRIBUTE_INPUT_TOKENS,
    GENERATION_ATTRIBUTE_OUTPUT_TOKENS,
    GENERATION_ATTRIBUTE_TOTAL_TOKENS,
    GENERATION_ATTRIBUTE_CONTENT,
    GENERATION_ATTRIBUTE_ROLE,
    GENERATION_ATTRIBUTE_STOP_REASON,
    GENERATION_ATTRIBUTE_FAILED,
    TOOL_ATTRIBUTE_NAME,
    TOOL_ATTRIBUTE_CALL_ID,
    TOOL_ATTRIBUTE_ARGUMENTS,
    TOOL_ATTRIBUTE_RESULT,
    TOOL_ATTRIBUTE_ERROR,
    TOOL_ATTRIBUTE_STOPPED,
    SCORER_ATTRIBUTE_NAME,
    SCORER_ATTRIBUTE_SCORE,
    SCORER_ATTRIBUTE_PASSED,
    SCORER_ATTRIBUTE_RATIONALE,
    SCORER_ATTRIBUTE_STEP,
    TRIAL_ATTRIBUTE_ID,
    TRIAL_ATTRIBUTE_STEP,
    TRIAL_ATTRIBUTE_CANDIDATE,
    TRIAL_ATTRIBUTE_IS_PROBE,
    TRIAL_ATTRIBUTE_STATUS,
    TRIAL_ATTRIBUTE_SCORES,
)


class TestGenerationSpan:
    """Tests for generation_span factory."""

    def test_basic_creation(self):
        """Test creating a basic generation span."""
        span = generation_span(step=1, model="gpt-4")

        # Check pre-attributes before entering context
        attrs = span._pre_attributes
        assert attrs.get(GENERATION_ATTRIBUTE_MODEL) == "gpt-4"
        assert attrs.get(GENERATION_ATTRIBUTE_INPUT_TOKENS) == 0
        assert attrs.get(GENERATION_ATTRIBUTE_OUTPUT_TOKENS) == 0
        assert attrs.get(GENERATION_ATTRIBUTE_TOTAL_TOKENS) == 0
        assert attrs.get(GENERATION_ATTRIBUTE_FAILED) is False

    def test_with_token_usage(self):
        """Test generation span with token usage."""
        span = generation_span(
            step=1,
            model="claude-3",
            input_tokens=100,
            output_tokens=50,
        )

        attrs = span._pre_attributes
        assert attrs.get(GENERATION_ATTRIBUTE_INPUT_TOKENS) == 100
        assert attrs.get(GENERATION_ATTRIBUTE_OUTPUT_TOKENS) == 50
        assert attrs.get(GENERATION_ATTRIBUTE_TOTAL_TOKENS) == 150

    def test_with_content_and_role(self):
        """Test generation span with content and role."""
        span = generation_span(
            step=1,
            content="Hello, world!",
            role="assistant",
        )

        attrs = span._pre_attributes
        assert attrs.get(GENERATION_ATTRIBUTE_CONTENT) == "Hello, world!"
        assert attrs.get(GENERATION_ATTRIBUTE_ROLE) == "assistant"

    def test_content_truncation(self):
        """Test that content is truncated to 4000 chars."""
        long_content = "x" * 5000
        span = generation_span(step=1, content=long_content)

        attrs = span._pre_attributes
        assert len(attrs.get(GENERATION_ATTRIBUTE_CONTENT)) == 4000

    def test_with_tool_calls(self):
        """Test generation span with tool calls."""
        import json

        tool_calls = [
            {"name": "get_weather", "id": "call_123", "arguments": {"city": "NYC"}}
        ]
        span = generation_span(step=1, tool_calls=tool_calls)

        attrs = span._pre_attributes
        # Tool calls are JSON-serialized
        assert json.loads(attrs.get("dreadnode.generation.tool_calls")) == tool_calls

    def test_with_stop_reason_and_failed(self):
        """Test generation span with stop reason and failed flag."""
        span = generation_span(
            step=1,
            stop_reason="max_tokens",
            failed=True,
        )

        attrs = span._pre_attributes
        assert attrs.get(GENERATION_ATTRIBUTE_STOP_REASON) == "max_tokens"
        assert attrs.get(GENERATION_ATTRIBUTE_FAILED) is True


class TestToolSpan:
    """Tests for tool_span factory."""

    def test_basic_creation(self):
        """Test creating a basic tool span."""
        span = tool_span("get_weather", call_id="call_123")

        attrs = span._pre_attributes
        assert attrs.get(TOOL_ATTRIBUTE_NAME) == "get_weather"
        assert attrs.get(TOOL_ATTRIBUTE_CALL_ID) == "call_123"
        assert attrs.get(TOOL_ATTRIBUTE_STOPPED) is False

    def test_with_dict_arguments(self):
        """Test tool span with dict arguments."""
        import json

        span = tool_span(
            "get_weather",
            call_id="call_123",
            arguments={"city": "NYC", "units": "celsius"},
        )

        attrs = span._pre_attributes
        # Arguments are JSON-serialized
        assert json.loads(attrs.get(TOOL_ATTRIBUTE_ARGUMENTS)) == {"city": "NYC", "units": "celsius"}

    def test_with_string_arguments(self):
        """Test tool span with JSON string arguments."""
        import json

        span = tool_span(
            "get_weather",
            call_id="call_123",
            arguments='{"city": "NYC"}',
        )

        attrs = span._pre_attributes
        # Should be parsed then re-serialized
        assert json.loads(attrs.get(TOOL_ATTRIBUTE_ARGUMENTS)) == {"city": "NYC"}

    def test_with_result(self):
        """Test tool span with result."""
        span = tool_span(
            "get_weather",
            call_id="call_123",
            result="The weather is sunny.",
        )

        attrs = span._pre_attributes
        assert attrs.get(TOOL_ATTRIBUTE_RESULT) == "The weather is sunny."

    def test_result_truncation(self):
        """Test that result is truncated to 4000 chars."""
        long_result = "x" * 5000
        span = tool_span("test", call_id="call_123", result=long_result)

        attrs = span._pre_attributes
        assert len(attrs.get(TOOL_ATTRIBUTE_RESULT)) == 4000

    def test_with_error(self):
        """Test tool span with error."""
        span = tool_span(
            "get_weather",
            call_id="call_123",
            error="Connection timeout",
        )

        attrs = span._pre_attributes
        assert attrs.get(TOOL_ATTRIBUTE_ERROR) == "Connection timeout"

    def test_error_truncation(self):
        """Test that error is truncated to 1000 chars."""
        long_error = "x" * 2000
        span = tool_span("test", call_id="call_123", error=long_error)

        attrs = span._pre_attributes
        assert len(attrs.get(TOOL_ATTRIBUTE_ERROR)) == 1000

    def test_with_stopped_flag(self):
        """Test tool span with stopped flag."""
        span = tool_span(
            "finish",
            call_id="call_123",
            stopped=True,
        )

        attrs = span._pre_attributes
        assert attrs.get(TOOL_ATTRIBUTE_STOPPED) is True


class TestScorerSpan:
    """Tests for scorer_span factory."""

    def test_basic_creation(self):
        """Test creating a basic scorer span."""
        span = scorer_span("quality")

        attrs = span._pre_attributes
        assert attrs.get(SCORER_ATTRIBUTE_NAME) == "quality"

    def test_with_score(self):
        """Test scorer span with score."""
        span = scorer_span("quality", score=0.85)

        attrs = span._pre_attributes
        assert attrs.get(SCORER_ATTRIBUTE_SCORE) == 0.85

    def test_with_passed(self):
        """Test scorer span with passed flag."""
        span = scorer_span("quality", score=0.85, passed=True)

        attrs = span._pre_attributes
        assert attrs.get(SCORER_ATTRIBUTE_PASSED) is True

    def test_with_rationale(self):
        """Test scorer span with rationale."""
        span = scorer_span(
            "quality",
            score=0.85,
            rationale="Good grammar and coherent structure.",
        )

        attrs = span._pre_attributes
        assert attrs.get(SCORER_ATTRIBUTE_RATIONALE) == "Good grammar and coherent structure."

    def test_rationale_truncation(self):
        """Test that rationale is truncated to 1000 chars."""
        long_rationale = "x" * 2000
        span = scorer_span("quality", rationale=long_rationale)

        attrs = span._pre_attributes
        assert len(attrs.get(SCORER_ATTRIBUTE_RATIONALE)) == 1000

    def test_with_step(self):
        """Test scorer span with step."""
        span = scorer_span("quality", step=5)

        attrs = span._pre_attributes
        assert attrs.get(SCORER_ATTRIBUTE_STEP) == 5


class TestTrialSpan:
    """Tests for trial_span factory."""

    def test_basic_creation(self):
        """Test creating a basic trial span."""
        span = trial_span(
            trial_id="trial_abc123",
            step=1,
            candidate={"learning_rate": 0.01},
        )

        attrs = span._pre_attributes
        assert attrs.get(TRIAL_ATTRIBUTE_ID) == "trial_abc123"
        assert attrs.get(TRIAL_ATTRIBUTE_STEP) == 1
        assert attrs.get(TRIAL_ATTRIBUTE_IS_PROBE) is False

    def test_with_task_name(self):
        """Test trial span with task name."""
        span = trial_span(
            trial_id="trial_123",
            step=1,
            candidate={"lr": 0.01},
            task_name="train_model",
        )

        # Label should include task name
        attrs = span._pre_attributes
        assert "train_model" in attrs.get("dreadnode.label", "")

    def test_with_is_probe(self):
        """Test trial span with is_probe flag."""
        span = trial_span(
            trial_id="probe_123",
            step=1,
            candidate={"lr": 0.01},
            is_probe=True,
        )

        attrs = span._pre_attributes
        assert attrs.get(TRIAL_ATTRIBUTE_IS_PROBE) is True
        # Tags should include "probe"
        assert "probe" in attrs.get("dreadnode.tags", ())

    def test_with_status(self):
        """Test trial span with status."""
        span = trial_span(
            trial_id="trial_123",
            step=1,
            candidate={"lr": 0.01},
            status="finished",
        )

        attrs = span._pre_attributes
        assert attrs.get(TRIAL_ATTRIBUTE_STATUS) == "finished"

    def test_with_scores(self):
        """Test trial span with scores."""
        import json

        scores = {"accuracy": 0.95, "f1": 0.92}
        span = trial_span(
            trial_id="trial_123",
            step=1,
            candidate={"lr": 0.01},
            scores=scores,
        )

        attrs = span._pre_attributes
        # Scores are JSON-serialized
        assert json.loads(attrs.get(TRIAL_ATTRIBUTE_SCORES)) == scores
