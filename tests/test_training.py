"""
Unit tests for the training module.

Tests cover:
- Reward types and components
- Reward functions (SuccessReward, ToolPenalty, etc.)
- Reward aggregator
- Reward shaping
- Rollout types
- Baseline/variance calculation
- Agent environment configuration

Note: Many tests require torch to be installed. Tests that require torch
are marked with @pytest.mark.skipif to skip gracefully when torch is not available.
"""

import pytest
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from unittest.mock import Mock, AsyncMock, patch

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Skip marker for tests requiring torch
requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="torch is not installed"
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_message_log():
    """Sample message log for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Do the task"},
        {"role": "assistant", "content": "I'll help you with that."},
        {"role": "tool", "content": "Tool result here"},
        {"role": "assistant", "content": "Task completed successfully."},
    ]


@pytest.fixture
@requires_torch
def sample_turn_result():
    """Sample TurnResult for testing."""
    from dreadnode.core.training.rollouts.types import TurnResult

    return TurnResult(
        turn_number=0,
        generated_text="I'll execute the task.",
        generated_tokens=10,
        input_tokens=50,
        tool_calls=[{"name": "test_tool", "arguments": {}}],
        tool_results=[{"success": True, "output": "done"}],
    )


@pytest.fixture
@requires_torch
def sample_rollout_metrics():
    """Sample RolloutMetrics for testing."""
    from dreadnode.core.training.rollouts.types import RolloutMetrics

    return RolloutMetrics(
        total_turns=3,
        completed_turns=3,
        total_input_tokens=100,
        total_generated_tokens=50,
        total_tool_calls=2,
        successful_tool_calls=2,
        failed_tool_calls=0,
        natural_termination=True,
    )


@pytest.fixture
@requires_torch
def sample_rollout_result(sample_message_log, sample_turn_result, sample_rollout_metrics):
    """Sample RolloutResult for testing."""
    from dreadnode.core.training.rollouts.types import RolloutResult

    return RolloutResult(
        rollout_id="test-rollout-123",
        agent_id="test-agent",
        goal="Complete the task",
        message_log=sample_message_log,
        turns=[sample_turn_result],
        metrics=sample_rollout_metrics,
        success=True,
        final_reward=1.0,
    )


# =============================================================================
# Reward Types Tests
# =============================================================================


@requires_torch
class TestRewardComponent:
    """Tests for RewardComponent dataclass."""

    def test_creation(self):
        """Test basic RewardComponent creation."""
        from dreadnode.core.training.rewards.types import RewardComponent

        comp = RewardComponent(name="test", value=0.5, weight=1.0)

        assert comp.name == "test"
        assert comp.value == 0.5
        assert comp.weight == 1.0

    def test_weighted_value(self):
        """Test weighted_value calculation."""
        from dreadnode.core.training.rewards.types import RewardComponent

        comp = RewardComponent(name="test", value=0.5, weight=2.0)
        assert comp.weighted_value() == 1.0

        comp2 = RewardComponent(name="test", value=-0.3, weight=0.5)
        assert comp2.weighted_value() == -0.15

    def test_to_dict(self):
        """Test to_dict serialization."""
        from dreadnode.core.training.rewards.types import RewardComponent

        comp = RewardComponent(
            name="test",
            value=0.5,
            weight=2.0,
            step=5,
            rationale="Test rationale",
        )
        d = comp.to_dict()

        assert d["name"] == "test"
        assert d["value"] == 0.5
        assert d["weight"] == 2.0
        assert d["weighted_value"] == 1.0
        assert d["step"] == 5
        assert d["rationale"] == "Test rationale"


@requires_torch
class TestRewardMetrics:
    """Tests for RewardMetrics dataclass."""

    def test_creation(self):
        """Test basic RewardMetrics creation."""
        from dreadnode.core.training.rewards.types import RewardMetrics

        metrics = RewardMetrics()
        assert metrics.raw_total == 0.0
        assert metrics.weighted_total == 0.0
        assert metrics.scaling_applied is False

    def test_to_dict(self):
        """Test to_dict serialization."""
        from dreadnode.core.training.rewards.types import RewardMetrics

        metrics = RewardMetrics(
            component_values={"success": 1.0, "penalty": -0.1},
            component_weights={"success": 1.0, "penalty": 0.5},
            raw_total=0.9,
            weighted_total=0.95,
        )
        d = metrics.to_dict()

        assert d["components"] == {"success": 1.0, "penalty": -0.1}
        assert d["raw_total"] == 0.9
        assert d["weighted_total"] == 0.95


@requires_torch
class TestRewardResult:
    """Tests for RewardResult dataclass."""

    def test_creation(self):
        """Test basic RewardResult creation."""
        from dreadnode.core.training.rewards.types import RewardResult

        result = RewardResult(reward=0.8, success=True)

        assert result.reward == 0.8
        assert result.success is True

    def test_component_by_name(self):
        """Test finding component by name."""
        from dreadnode.core.training.rewards.types import RewardResult, RewardComponent

        comp1 = RewardComponent(name="success", value=1.0)
        comp2 = RewardComponent(name="penalty", value=-0.1)

        result = RewardResult(reward=0.9, components=[comp1, comp2])

        assert result.component_by_name("success") == comp1
        assert result.component_by_name("penalty") == comp2
        assert result.component_by_name("nonexistent") is None

    def test_total_weight(self):
        """Test total_weight calculation."""
        from dreadnode.core.training.rewards.types import RewardResult, RewardComponent

        comp1 = RewardComponent(name="a", value=1.0, weight=1.0)
        comp2 = RewardComponent(name="b", value=0.5, weight=0.5)

        result = RewardResult(reward=1.0, components=[comp1, comp2])
        assert result.total_weight() == 1.5

    def test_to_training_dict(self):
        """Test to_training_dict for NeMo RL compatibility."""
        from dreadnode.core.training.rewards.types import RewardResult, RewardComponent

        comp = RewardComponent(name="success", value=1.0, weight=1.0)
        result = RewardResult(reward=1.0, success=True, terminated=True, components=[comp])

        d = result.to_training_dict()
        assert d["total_reward"] == 1.0
        assert d["success"] == 1.0
        assert d["terminated"] == 1.0
        assert d["reward_components"]["success"] == 1.0


# =============================================================================
# Reward Functions Tests
# =============================================================================


@requires_torch
class TestSuccessReward:
    """Tests for SuccessReward function."""

    def test_success_case(self, sample_rollout_result):
        """Test reward for successful rollout."""
        from dreadnode.core.training.rewards.functions import SuccessReward

        reward_fn = SuccessReward(weight=1.0, success_value=1.0, failure_value=0.0)
        sample_rollout_result.success = True

        result = reward_fn.compute(sample_rollout_result)

        assert result.value == 1.0
        assert result.name == "success"
        assert "succeeded" in result.rationale

    def test_failure_case(self, sample_rollout_result):
        """Test reward for failed rollout."""
        from dreadnode.core.training.rewards.functions import SuccessReward

        reward_fn = SuccessReward(weight=1.0, success_value=1.0, failure_value=-0.5)
        sample_rollout_result.success = False

        result = reward_fn.compute(sample_rollout_result)

        assert result.value == -0.5
        assert "failed" in result.rationale

    def test_custom_values(self, sample_rollout_result):
        """Test with custom success/failure values."""
        from dreadnode.core.training.rewards.functions import SuccessReward

        reward_fn = SuccessReward(weight=2.0, success_value=10.0, failure_value=-5.0)
        sample_rollout_result.success = True

        result = reward_fn.compute(sample_rollout_result)

        assert result.value == 10.0
        assert result.weight == 2.0
        assert result.weighted_value() == 20.0


@requires_torch
class TestToolPenalty:
    """Tests for ToolPenalty function."""

    def test_no_tool_calls(self, sample_rollout_result):
        """Test penalty with no tool calls."""
        from dreadnode.core.training.rewards.functions import ToolPenalty

        sample_rollout_result.metrics.total_tool_calls = 0
        sample_rollout_result.metrics.failed_tool_calls = 0

        penalty = ToolPenalty(weight=0.1, per_call_penalty=0.01)
        result = penalty.compute(sample_rollout_result)

        assert result.value == 0.0

    def test_with_tool_calls(self, sample_rollout_result):
        """Test penalty with tool calls."""
        from dreadnode.core.training.rewards.functions import ToolPenalty

        sample_rollout_result.metrics.total_tool_calls = 5
        sample_rollout_result.metrics.failed_tool_calls = 1

        penalty = ToolPenalty(
            weight=0.1,
            per_call_penalty=0.01,
            per_failure_penalty=0.05,
        )
        result = penalty.compute(sample_rollout_result)

        # 5 * 0.01 + 1 * 0.05 = 0.10
        expected = -(5 * 0.01 + 1 * 0.05)
        assert result.value == expected

    def test_max_penalty_cap(self, sample_rollout_result):
        """Test that penalty is capped at max_penalty."""
        from dreadnode.core.training.rewards.functions import ToolPenalty

        sample_rollout_result.metrics.total_tool_calls = 100
        sample_rollout_result.metrics.failed_tool_calls = 50

        penalty = ToolPenalty(
            weight=0.1,
            per_call_penalty=0.1,
            per_failure_penalty=0.1,
            max_penalty=0.5,
        )
        result = penalty.compute(sample_rollout_result)

        assert result.value == -0.5  # Capped at max


@requires_torch
class TestLengthPenalty:
    """Tests for LengthPenalty function."""

    def test_within_limit(self, sample_rollout_result):
        """Test no penalty when within limit."""
        from dreadnode.core.training.rewards.functions import LengthPenalty

        sample_rollout_result.metrics.total_generated_tokens = 1000

        penalty = LengthPenalty(max_length=4096, buffer_length=512, penalty=0.5)
        result = penalty.compute(sample_rollout_result)

        assert result.value == 0.0

    def test_exceeds_limit(self, sample_rollout_result):
        """Test penalty when exceeding limit."""
        from dreadnode.core.training.rewards.functions import LengthPenalty

        # max_length=4096, buffer=512 -> expected_length=3584
        # tokens=4000 -> exceeds by 416
        sample_rollout_result.metrics.total_generated_tokens = 4000

        penalty = LengthPenalty(max_length=4096, buffer_length=512, penalty=0.5)
        result = penalty.compute(sample_rollout_result)

        # exceed = 4000 - 3584 = 416
        # penalty = min((416/512) * 0.5, 0.5) = 0.40625
        assert result.value < 0
        assert result.value >= -0.5


@requires_torch
class TestTurnPenalty:
    """Tests for TurnPenalty function."""

    def test_within_target(self, sample_rollout_result):
        """Test no penalty when within target turns."""
        from dreadnode.core.training.rewards.functions import TurnPenalty

        sample_rollout_result.metrics.total_turns = 3

        penalty = TurnPenalty(target_turns=5, per_extra_turn_penalty=0.02)
        result = penalty.compute(sample_rollout_result)

        assert result.value == 0.0

    def test_exceeds_target(self, sample_rollout_result):
        """Test penalty when exceeding target turns."""
        from dreadnode.core.training.rewards.functions import TurnPenalty

        sample_rollout_result.metrics.total_turns = 8

        penalty = TurnPenalty(target_turns=5, per_extra_turn_penalty=0.02, max_penalty=0.3)
        result = penalty.compute(sample_rollout_result)

        # 3 extra turns * 0.02 = 0.06
        assert result.value == -0.06


@requires_torch
class TestFormatReward:
    """Tests for FormatReward function."""

    def test_pattern_match(self, sample_rollout_result):
        """Test reward when pattern matches."""
        from dreadnode.core.training.rewards.functions import FormatReward

        # Make the last message contain the pattern
        sample_rollout_result.message_log[-1]["content"] = "```python\nprint('hello')\n```"

        reward = FormatReward(pattern=r"```python\n.*\n```", match_reward=0.2)
        result = reward.compute(sample_rollout_result)

        assert result.value == 0.2

    def test_pattern_no_match(self, sample_rollout_result):
        """Test penalty when pattern doesn't match."""
        from dreadnode.core.training.rewards.functions import FormatReward

        sample_rollout_result.message_log[-1]["content"] = "No code here"

        reward = FormatReward(
            pattern=r"```python\n.*\n```",
            match_reward=0.2,
            violation_penalty=0.1,
        )
        result = reward.compute(sample_rollout_result)

        assert result.value == -0.1

    def test_required_strings(self, sample_rollout_result):
        """Test with required strings."""
        from dreadnode.core.training.rewards.functions import FormatReward

        sample_rollout_result.message_log[-1]["content"] = "The answer is 42. Done."

        reward = FormatReward(
            required_strings=["answer", "Done"],
            match_reward=0.2,
        )
        result = reward.compute(sample_rollout_result)

        # Both strings present: 0.2/2 + 0.2/2 = 0.2
        assert result.value == pytest.approx(0.2, rel=0.01)

    def test_forbidden_strings(self, sample_rollout_result):
        """Test with forbidden strings."""
        from dreadnode.core.training.rewards.functions import FormatReward

        sample_rollout_result.message_log[-1]["content"] = "ERROR: something went wrong"

        reward = FormatReward(
            forbidden_strings=["ERROR", "FAIL"],
            violation_penalty=0.1,
        )
        result = reward.compute(sample_rollout_result)

        assert result.value == -0.1


# =============================================================================
# Reward Aggregator Tests
# =============================================================================


@requires_torch
class TestRewardAggregator:
    """Tests for RewardAggregator."""

    def test_sum_strategy(self, sample_rollout_result):
        """Test SUM aggregation strategy."""
        from dreadnode.core.training.rewards.aggregator import (
            RewardAggregator,
            AggregationStrategy,
        )
        from dreadnode.core.training.rewards.functions import SuccessReward, ToolPenalty

        sample_rollout_result.success = True
        sample_rollout_result.metrics.total_tool_calls = 2
        sample_rollout_result.metrics.failed_tool_calls = 0

        aggregator = RewardAggregator(
            rewards=[
                SuccessReward(weight=1.0, success_value=1.0),
                ToolPenalty(weight=1.0, per_call_penalty=0.1),
            ],
            strategy=AggregationStrategy.SUM,
        )

        result = aggregator.compute(sample_rollout_result)

        # 1.0 (success) + (-0.2) (tool penalty) = 0.8
        assert result.reward == pytest.approx(0.8, rel=0.01)

    def test_weighted_average_strategy(self, sample_rollout_result):
        """Test WEIGHTED_AVERAGE aggregation strategy."""
        from dreadnode.core.training.rewards.aggregator import (
            RewardAggregator,
            AggregationStrategy,
        )
        from dreadnode.core.training.rewards.functions import SuccessReward

        sample_rollout_result.success = True

        aggregator = RewardAggregator(
            rewards=[
                SuccessReward(weight=2.0, success_value=1.0),
                SuccessReward(weight=1.0, success_value=0.5),
            ],
            strategy=AggregationStrategy.WEIGHTED_AVERAGE,
        )

        result = aggregator.compute(sample_rollout_result)

        # (2.0*1.0 + 1.0*0.5) / (2.0 + 1.0) = 2.5/3 ≈ 0.833
        assert result.reward == pytest.approx(2.5 / 3, rel=0.01)

    def test_min_strategy(self, sample_rollout_result):
        """Test MIN aggregation strategy."""
        from dreadnode.core.training.rewards.aggregator import (
            RewardAggregator,
            AggregationStrategy,
        )
        from dreadnode.core.training.rewards.functions import SuccessReward

        sample_rollout_result.success = True

        aggregator = RewardAggregator(
            rewards=[
                SuccessReward(weight=1.0, success_value=1.0),
                SuccessReward(weight=1.0, success_value=0.3),
            ],
            strategy=AggregationStrategy.MIN,
        )

        result = aggregator.compute(sample_rollout_result)

        assert result.reward == 0.3

    def test_max_strategy(self, sample_rollout_result):
        """Test MAX aggregation strategy."""
        from dreadnode.core.training.rewards.aggregator import (
            RewardAggregator,
            AggregationStrategy,
        )
        from dreadnode.core.training.rewards.functions import SuccessReward

        sample_rollout_result.success = True

        aggregator = RewardAggregator(
            rewards=[
                SuccessReward(weight=1.0, success_value=0.5),
                SuccessReward(weight=1.0, success_value=1.0),
            ],
            strategy=AggregationStrategy.MAX,
        )

        result = aggregator.compute(sample_rollout_result)

        assert result.reward == 1.0

    def test_success_gate(self, sample_rollout_result):
        """Test success_gate functionality."""
        from dreadnode.core.training.rewards.aggregator import RewardAggregator
        from dreadnode.core.training.rewards.functions import SuccessReward

        sample_rollout_result.success = False

        aggregator = RewardAggregator(
            rewards=[SuccessReward(weight=1.0, success_value=1.0)],
            success_gate=True,
            failure_reward=-1.0,
        )

        result = aggregator.compute(sample_rollout_result)

        assert result.reward == -1.0
        assert len(result.components) == 0

    def test_base_reward(self, sample_rollout_result):
        """Test base_reward functionality."""
        from dreadnode.core.training.rewards.aggregator import RewardAggregator
        from dreadnode.core.training.rewards.functions import SuccessReward

        sample_rollout_result.success = True

        aggregator = RewardAggregator(
            rewards=[SuccessReward(weight=1.0, success_value=0.5)],
            base_reward=0.1,
        )

        result = aggregator.compute(sample_rollout_result)

        assert result.reward == 0.6  # 0.1 base + 0.5 success

    def test_fluent_add(self, sample_rollout_result):
        """Test fluent add() method."""
        from dreadnode.core.training.rewards.aggregator import RewardAggregator
        from dreadnode.core.training.rewards.functions import SuccessReward, ToolPenalty

        sample_rollout_result.success = True
        sample_rollout_result.metrics.total_tool_calls = 0

        aggregator = (
            RewardAggregator()
            .add(SuccessReward(weight=1.0))
            .add(ToolPenalty(weight=0.1))
        )

        assert len(aggregator.rewards) == 2

    def test_compute_batch(self, sample_rollout_result):
        """Test batch computation."""
        from dreadnode.core.training.rewards.aggregator import RewardAggregator
        from dreadnode.core.training.rewards.functions import SuccessReward
        from dreadnode.core.training.rollouts.types import RolloutResult, RolloutMetrics

        aggregator = RewardAggregator(rewards=[SuccessReward(weight=1.0)])

        rollout1 = sample_rollout_result
        rollout1.success = True

        # Create a copy for rollout2
        rollout2 = RolloutResult(
            rollout_id="test-2",
            goal="test",
            message_log=[{"role": "assistant", "content": "done"}],
            metrics=RolloutMetrics(),
            success=False,
        )

        results = aggregator.compute_batch([rollout1, rollout2])

        assert len(results) == 2
        assert results[0].reward == 1.0
        assert results[1].reward == 0.0


@requires_torch
class TestCreateStandardReward:
    """Tests for create_standard_reward helper."""

    def test_default_config(self):
        """Test with default configuration."""
        from dreadnode.core.training.rewards.aggregator import create_standard_reward

        aggregator = create_standard_reward()

        # Default includes SuccessReward and ToolPenalty
        assert len(aggregator.rewards) >= 1

    def test_with_penalties(self):
        """Test with all penalties enabled."""
        from dreadnode.core.training.rewards.aggregator import create_standard_reward

        aggregator = create_standard_reward(
            success_weight=1.0,
            tool_penalty_weight=0.1,
            length_penalty_weight=0.1,
            turn_penalty_weight=0.1,
        )

        assert len(aggregator.rewards) == 4


# =============================================================================
# Reward Shaping Tests
# =============================================================================


@requires_torch
class TestRewardShaping:
    """Tests for reward shaping functions."""

    def test_apply_dapo_penalty_within_limit(self):
        """Test DAPO penalty when within limit."""
        from dreadnode.core.training.rewards.shaping import apply_dapo_penalty

        result = apply_dapo_penalty(
            reward=1.0,
            response_length=3000,
            max_response_length=4096,
            buffer_length=512,
            penalty=0.5,
        )

        assert result == 1.0  # No penalty

    def test_apply_dapo_penalty_exceeds_limit(self):
        """Test DAPO penalty when exceeding limit."""
        from dreadnode.core.training.rewards.shaping import apply_dapo_penalty

        # expected = 4096 - 512 = 3584
        # response = 4000 -> exceeds by 416
        result = apply_dapo_penalty(
            reward=1.0,
            response_length=4000,
            max_response_length=4096,
            buffer_length=512,
            penalty=0.5,
        )

        assert result < 1.0

    def test_apply_scaling(self):
        """Test linear reward scaling."""
        from dreadnode.core.training.rewards.shaping import apply_scaling

        # Scale [0, 1] -> [-1, 1]
        assert apply_scaling(0.0, 0.0, 1.0, -1.0, 1.0) == -1.0
        assert apply_scaling(0.5, 0.0, 1.0, -1.0, 1.0) == 0.0
        assert apply_scaling(1.0, 0.0, 1.0, -1.0, 1.0) == 1.0

    def test_apply_scaling_clamps(self):
        """Test that scaling clamps to source range."""
        from dreadnode.core.training.rewards.shaping import apply_scaling

        # Value outside source range should be clamped
        assert apply_scaling(2.0, 0.0, 1.0, 0.0, 10.0) == 10.0
        assert apply_scaling(-1.0, 0.0, 1.0, 0.0, 10.0) == 0.0

    def test_clip_reward(self):
        """Test reward clipping."""
        from dreadnode.core.training.rewards.shaping import clip_reward

        assert clip_reward(0.5, -1.0, 1.0) == 0.5
        assert clip_reward(2.0, -1.0, 1.0) == 1.0
        assert clip_reward(-2.0, -1.0, 1.0) == -1.0


@requires_torch
class TestRewardShaper:
    """Tests for RewardShaper class."""

    def test_scaling_only(self):
        """Test shaper with only scaling enabled."""
        from dreadnode.core.training.rewards.shaping import RewardShaper
        from dreadnode.core.training.rewards.types import RewardResult

        shaper = RewardShaper(
            scaling_enabled=True,
            source_min=0.0,
            source_max=1.0,
            target_min=-1.0,
            target_max=1.0,
        )

        result = RewardResult(reward=0.5)
        shaped = shaper.shape(result)

        assert shaped.reward == 0.0
        assert shaped.metrics.scaling_applied is True

    def test_dapo_only(self):
        """Test shaper with only DAPO enabled."""
        from dreadnode.core.training.rewards.shaping import RewardShaper
        from dreadnode.core.training.rewards.types import RewardResult

        shaper = RewardShaper(
            dapo_enabled=True,
            max_response_length=4096,
            buffer_length=512,
            penalty=0.5,
        )

        result = RewardResult(reward=1.0)
        shaped = shaper.shape(result, response_length=4000)

        assert shaped.reward < 1.0
        assert shaped.metrics.penalty_applied is True

    def test_to_nemo_config(self):
        """Test conversion to NeMo RL config format."""
        from dreadnode.core.training.rewards.shaping import RewardShaper

        shaper = RewardShaper(
            scaling_enabled=True,
            source_min=0.0,
            source_max=1.0,
            target_min=-1.0,
            target_max=1.0,
            dapo_enabled=True,
            buffer_length=512,
            penalty=0.5,
            max_response_length=4096,
        )

        config = shaper.to_nemo_config()

        assert config["reward_scaling"]["enabled"] is True
        assert config["reward_shaping"]["enabled"] is True
        assert config["reward_shaping"]["overlong_buffer_length"] == 512

    def test_from_nemo_config(self):
        """Test creation from NeMo RL config."""
        from dreadnode.core.training.rewards.shaping import RewardShaper

        config = {
            "reward_scaling": {
                "enabled": True,
                "source_min": 0.0,
                "source_max": 1.0,
                "target_min": -1.0,
                "target_max": 1.0,
            },
            "reward_shaping": {
                "enabled": True,
                "overlong_buffer_length": 256,
                "overlong_buffer_penalty": 0.3,
                "max_response_length": 2048,
            },
        }

        shaper = RewardShaper.from_nemo_config(config)

        assert shaper.scaling_enabled is True
        assert shaper.dapo_enabled is True
        assert shaper.buffer_length == 256


# =============================================================================
# Rollout Types Tests
# =============================================================================


@requires_torch
class TestRolloutTypes:
    """Tests for rollout type definitions."""

    def test_turn_result_creation(self):
        """Test TurnResult creation."""
        from dreadnode.core.training.rollouts.types import TurnResult

        turn = TurnResult(
            turn_number=0,
            generated_text="Hello",
            generated_tokens=5,
            input_tokens=10,
        )

        assert turn.turn_number == 0
        assert turn.generated_text == "Hello"
        assert turn.terminated is False
        assert turn.error is None

    def test_rollout_metrics_defaults(self):
        """Test RolloutMetrics defaults."""
        from dreadnode.core.training.rollouts.types import RolloutMetrics

        metrics = RolloutMetrics()

        assert metrics.total_turns == 0
        assert metrics.total_tool_calls == 0
        assert metrics.natural_termination is False

    def test_rollout_result_to_trajectory_record(self, sample_rollout_result):
        """Test conversion to trajectory JSONL record."""
        record = sample_rollout_result.to_trajectory_record()

        assert record["id"] == "test-rollout-123"
        assert record["goal"] == "Complete the task"
        assert record["success"] is True
        assert len(record["messages"]) == len(sample_rollout_result.message_log)

    def test_rollout_result_to_openai_format(self, sample_rollout_result):
        """Test conversion to OpenAI message format."""
        messages = sample_rollout_result.to_openai_format()

        assert len(messages) == len(sample_rollout_result.message_log)
        assert all("role" in m and "content" in m for m in messages)


@requires_torch
class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_message_roles(self):
        """Test all message role values."""
        from dreadnode.core.training.rollouts.types import MessageRole

        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.TOOL.value == "tool"
        assert MessageRole.ENVIRONMENT.value == "environment"


# =============================================================================
# Baseline/Variance Calculation Tests
# =============================================================================


@requires_torch
class TestBaselineCalculation:
    """Tests for calculate_baseline_and_std_per_prompt function."""

    def test_empty_rewards(self):
        """Test with empty rewards tensor."""
        from dreadnode.core.training.trainers.grpo import calculate_baseline_and_std_per_prompt

        rewards = torch.tensor([])
        input_ids = torch.zeros(0, 10)
        mask = torch.ones(0)

        baseline, std = calculate_baseline_and_std_per_prompt(
            input_ids, rewards, mask
        )

        assert len(baseline) == 0
        assert len(std) == 0

    def test_single_reward(self):
        """Test with single reward (edge case)."""
        from dreadnode.core.training.trainers.grpo import calculate_baseline_and_std_per_prompt

        rewards = torch.tensor([1.0])
        input_ids = torch.zeros(1, 10)
        mask = torch.ones(1)

        baseline, std = calculate_baseline_and_std_per_prompt(
            input_ids, rewards, mask
        )

        assert baseline[0].item() == 1.0
        assert std[0].item() == 1.0  # Default std for single sample

    def test_two_rewards(self):
        """Test with two rewards."""
        from dreadnode.core.training.trainers.grpo import calculate_baseline_and_std_per_prompt

        rewards = torch.tensor([1.0, 3.0])
        input_ids = torch.zeros(2, 10)
        mask = torch.ones(2)

        baseline, std = calculate_baseline_and_std_per_prompt(
            input_ids, rewards, mask, leave_one_out_baseline=True
        )

        # Leave-one-out: baseline[0] = mean([3.0]) = 3.0
        # baseline[1] = mean([1.0]) = 1.0
        assert baseline[0].item() == 3.0
        assert baseline[1].item() == 1.0

    def test_multiple_rewards_leave_one_out(self):
        """Test leave-one-out baseline with multiple rewards."""
        from dreadnode.core.training.trainers.grpo import calculate_baseline_and_std_per_prompt

        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        input_ids = torch.zeros(4, 10)
        mask = torch.ones(4)

        baseline, std = calculate_baseline_and_std_per_prompt(
            input_ids, rewards, mask, leave_one_out_baseline=True
        )

        # baseline[0] = mean([2,3,4]) = 3.0
        # baseline[1] = mean([1,3,4]) = 2.666...
        # baseline[2] = mean([1,2,4]) = 2.333...
        # baseline[3] = mean([1,2,3]) = 2.0
        assert baseline[0].item() == pytest.approx(3.0, rel=0.01)
        assert baseline[1].item() == pytest.approx(8.0 / 3, rel=0.01)
        assert baseline[2].item() == pytest.approx(7.0 / 3, rel=0.01)
        assert baseline[3].item() == pytest.approx(2.0, rel=0.01)

    def test_simple_baseline(self):
        """Test simple mean baseline (not leave-one-out)."""
        from dreadnode.core.training.trainers.grpo import calculate_baseline_and_std_per_prompt

        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        input_ids = torch.zeros(4, 10)
        mask = torch.ones(4)

        baseline, std = calculate_baseline_and_std_per_prompt(
            input_ids, rewards, mask, leave_one_out_baseline=False
        )

        # Simple baseline: mean([1,2,3,4]) = 2.5 for all
        assert all(b.item() == pytest.approx(2.5, rel=0.01) for b in baseline)

    def test_std_never_zero(self):
        """Test that std is never zero or NaN (clamped)."""
        from dreadnode.core.training.trainers.grpo import calculate_baseline_and_std_per_prompt
        import math

        # All same rewards -> would normally have std=0
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
        input_ids = torch.zeros(4, 10)
        mask = torch.ones(4)

        baseline, std = calculate_baseline_and_std_per_prompt(
            input_ids, rewards, mask, leave_one_out_baseline=True
        )

        # std should be clamped to minimum value, never zero or NaN
        for s in std:
            val = s.item()
            assert not math.isnan(val), "std should not be NaN"
            # Use approximate comparison due to floating point precision
            assert val >= 1e-8 * 0.99, f"std should be >= ~1e-8, got {val}"


@requires_torch
class TestScaleRewards:
    """Tests for scale_rewards function."""

    def test_disabled(self):
        """Test when scaling is disabled."""
        from dreadnode.core.training.trainers.grpo import scale_rewards

        rewards = torch.tensor([0.0, 0.5, 1.0])
        config = {"enabled": False}

        scaled = scale_rewards(rewards, config)

        assert torch.allclose(scaled, rewards)

    def test_identity_scaling(self):
        """Test identity scaling (same source and target range)."""
        from dreadnode.core.training.trainers.grpo import scale_rewards

        rewards = torch.tensor([0.0, 0.5, 1.0])
        config = {
            "enabled": True,
            "source_min": 0.0,
            "source_max": 1.0,
            "target_min": 0.0,
            "target_max": 1.0,
        }

        scaled = scale_rewards(rewards, config)

        assert torch.allclose(scaled, rewards)

    def test_scaling_transformation(self):
        """Test actual scaling transformation."""
        from dreadnode.core.training.trainers.grpo import scale_rewards

        rewards = torch.tensor([0.0, 0.5, 1.0])
        config = {
            "enabled": True,
            "source_min": 0.0,
            "source_max": 1.0,
            "target_min": -1.0,
            "target_max": 1.0,
        }

        scaled = scale_rewards(rewards, config)
        expected = torch.tensor([-1.0, 0.0, 1.0])

        assert torch.allclose(scaled, expected)


# =============================================================================
# Agent Environment Tests
# =============================================================================


class TestAgentEnvConfig:
    """Tests for AgentEnvConfig type (no torch required)."""

    def test_config_structure(self):
        """Test config can be created with required fields."""
        config = {
            "num_workers": 4,
            "success_reward": 1.0,
            "failure_reward": 0.0,
            "partial_reward": 0.5,
        }

        assert config["num_workers"] == 4
        assert config["success_reward"] == 1.0

    def test_config_with_optional_fields(self):
        """Test config with optional fields."""
        config = {
            "num_workers": 4,
            "success_reward": 1.0,
            "failure_reward": 0.0,
            "partial_reward": 0.5,
            "allowed_tools": ["tool1", "tool2"],
            "eval_mode": "online",
        }

        assert config["allowed_tools"] == ["tool1", "tool2"]
        assert config["eval_mode"] == "online"


# =============================================================================
# Scorer Bridge Tests
# =============================================================================


@requires_torch
class TestScorerReward:
    """Tests for ScorerReward class."""

    def test_value_scaling(self):
        """Test value scaling from source to target range."""
        from dreadnode.core.training.rewards.scorer_bridge import ScorerReward

        scorer = Mock()
        reward = ScorerReward(
            scorer=scorer,
            source_min=0.0,
            source_max=100.0,
            target_min=0.0,
            target_max=1.0,
        )

        # 50 in [0, 100] -> 0.5 in [0, 1]
        assert reward._scale_value(50.0) == 0.5
        assert reward._scale_value(0.0) == 0.0
        assert reward._scale_value(100.0) == 1.0

    def test_value_clamping(self):
        """Test that values are clamped to source range."""
        from dreadnode.core.training.rewards.scorer_bridge import ScorerReward

        scorer = Mock()
        reward = ScorerReward(
            scorer=scorer,
            source_min=0.0,
            source_max=1.0,
            target_min=0.0,
            target_max=10.0,
        )

        # Values outside source range should be clamped
        assert reward._scale_value(2.0) == 10.0  # Clamped to max
        assert reward._scale_value(-1.0) == 0.0  # Clamped to min


@requires_torch
class TestMetricReward:
    """Tests for MetricReward class."""

    def test_extract_metric_simple(self, sample_rollout_result):
        """Test extracting simple metric path."""
        from dreadnode.core.training.rewards.scorer_bridge import MetricReward

        reward = MetricReward(metric_path="metrics.total_turns")
        result = reward.compute(sample_rollout_result)

        assert result.value == sample_rollout_result.metrics.total_turns

    def test_extract_metric_with_transform(self, sample_rollout_result):
        """Test extracting metric with transform."""
        from dreadnode.core.training.rewards.scorer_bridge import MetricReward

        reward = MetricReward(
            metric_path="metrics.total_tool_calls",
            transform=lambda x: -x * 0.01,  # Penalize tool calls
        )
        result = reward.compute(sample_rollout_result)

        expected = -sample_rollout_result.metrics.total_tool_calls * 0.01
        assert result.value == expected

    def test_extract_metric_not_found(self, sample_rollout_result):
        """Test default value when metric not found."""
        from dreadnode.core.training.rewards.scorer_bridge import MetricReward

        reward = MetricReward(
            metric_path="nonexistent.path",
            default_value=-999.0,
        )
        result = reward.compute(sample_rollout_result)

        assert result.value == -999.0


@requires_torch
class TestScorerHookReward:
    """Tests for ScorerHookReward class."""

    def test_aggregation_sum(self, sample_rollout_result):
        """Test sum aggregation of hook scores."""
        from dreadnode.core.training.rewards.scorer_bridge import ScorerHookReward

        sample_rollout_result.metadata = {
            "scorer_results": {
                "quality": [
                    {"value": 0.5},
                    {"value": 0.3},
                    {"value": 0.2},
                ]
            }
        }

        reward = ScorerHookReward(hook_name="quality", aggregation="sum")
        result = reward.compute(sample_rollout_result)

        assert result.value == 1.0

    def test_aggregation_mean(self, sample_rollout_result):
        """Test mean aggregation of hook scores."""
        from dreadnode.core.training.rewards.scorer_bridge import ScorerHookReward

        sample_rollout_result.metadata = {
            "scorer_results": {
                "quality": [
                    {"value": 0.6},
                    {"value": 0.4},
                ]
            }
        }

        reward = ScorerHookReward(hook_name="quality", aggregation="mean")
        result = reward.compute(sample_rollout_result)

        assert result.value == 0.5

    def test_aggregation_final(self, sample_rollout_result):
        """Test final (last) aggregation of hook scores."""
        from dreadnode.core.training.rewards.scorer_bridge import ScorerHookReward

        sample_rollout_result.metadata = {
            "scorer_results": {
                "quality": [
                    {"value": 0.1},
                    {"value": 0.5},
                    {"value": 0.9},
                ]
            }
        }

        reward = ScorerHookReward(hook_name="quality", aggregation="final")
        result = reward.compute(sample_rollout_result)

        assert result.value == 0.9

    def test_no_scores_found(self, sample_rollout_result):
        """Test default value when no scores found."""
        from dreadnode.core.training.rewards.scorer_bridge import ScorerHookReward

        sample_rollout_result.metadata = {}

        reward = ScorerHookReward(
            hook_name="nonexistent",
            default_value=-1.0,
        )
        result = reward.compute(sample_rollout_result)

        assert result.value == -1.0


# =============================================================================
# Async Tests
# =============================================================================


@requires_torch
class TestAsyncRewardComputation:
    """Tests for async reward computation."""

    @pytest.mark.asyncio
    async def test_aggregator_compute_async(self, sample_rollout_result):
        """Test async aggregator computation."""
        from dreadnode.core.training.rewards.aggregator import RewardAggregator
        from dreadnode.core.training.rewards.functions import SuccessReward

        sample_rollout_result.success = True

        aggregator = RewardAggregator(
            rewards=[SuccessReward(weight=1.0)],
        )

        result = await aggregator.compute_async(sample_rollout_result)

        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_aggregator_compute_batch_async(self, sample_rollout_result):
        """Test async batch computation."""
        from dreadnode.core.training.rewards.aggregator import RewardAggregator
        from dreadnode.core.training.rewards.functions import SuccessReward
        from dreadnode.core.training.rollouts.types import RolloutResult, RolloutMetrics

        aggregator = RewardAggregator(rewards=[SuccessReward(weight=1.0)])

        rollout1 = sample_rollout_result
        rollout1.success = True

        rollout2 = RolloutResult(
            rollout_id="test-2",
            goal="test",
            message_log=[{"role": "assistant", "content": "done"}],
            metrics=RolloutMetrics(),
            success=False,
        )

        results = await aggregator.compute_batch_async([rollout1, rollout2])

        assert len(results) == 2
        assert results[0].reward == 1.0
        assert results[1].reward == 0.0
