"""
Reward functions for agent training.

Provides RewardFunction protocol and built-in reward implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
from typing import Any, Protocol, runtime_checkable

from dreadnode.core.training.rewards.types import RewardComponent
from dreadnode.core.training.rollouts.types import RolloutResult


@runtime_checkable
class RewardFunction(Protocol):
    """
    Protocol for reward computation.

    RewardFunctions take a RolloutResult and produce a RewardComponent.
    This is the bridge between agent execution and training.
    """

    name: str
    weight: float

    def compute(self, rollout: RolloutResult) -> RewardComponent:
        """
        Compute reward from a rollout.

        Args:
            rollout: Complete rollout result with message log and metrics.

        Returns:
            RewardComponent with the computed value.
        """
        ...

    async def compute_async(self, rollout: RolloutResult) -> RewardComponent:
        """
        Async version for scorers that need async evaluation.

        Default implementation wraps sync compute().
        """
        ...


class BaseRewardFunction(ABC):
    """Base class for reward functions."""

    def __init__(self, name: str | None = None, weight: float = 1.0):
        self.name = name or self.__class__.__name__
        self.weight = weight

    @abstractmethod
    def compute(self, rollout: RolloutResult) -> RewardComponent:
        """Compute reward from rollout."""
        ...

    async def compute_async(self, rollout: RolloutResult) -> RewardComponent:
        """Async compute - default wraps sync."""
        return self.compute(rollout)

    def _make_component(
        self,
        value: float,
        step: int | None = None,
        rationale: str | None = None,
        **attrs: Any,
    ) -> RewardComponent:
        """Helper to create a RewardComponent."""
        return RewardComponent(
            name=self.name,
            value=value,
            weight=self.weight,
            step=step,
            rationale=rationale,
            attributes=attrs,
        )


@dataclass
class SuccessReward(BaseRewardFunction):
    """
    Binary reward for task success.

    The most basic reward signal: 1.0 if task succeeded, 0.0 otherwise.

    Example:
        reward = SuccessReward(weight=1.0)
        component = reward.compute(rollout)
        # Returns 1.0 if rollout.success is True
    """

    weight: float = 1.0
    success_value: float = 1.0
    failure_value: float = 0.0

    def __post_init__(self):
        self.name = "success"

    def compute(self, rollout: RolloutResult) -> RewardComponent:
        """Return success_value if successful, else failure_value."""
        value = self.success_value if rollout.success else self.failure_value
        return self._make_component(
            value=value,
            rationale=f"Task {'succeeded' if rollout.success else 'failed'}",
            success=rollout.success,
        )


@dataclass
class ToolPenalty(BaseRewardFunction):
    """
    Penalty based on tool usage.

    Can penalize for:
    - Total tool calls (efficiency penalty)
    - Failed tool calls (error penalty)
    - Specific tool types

    Example:
        penalty = ToolPenalty(
            weight=0.1,
            per_call_penalty=0.01,
            per_failure_penalty=0.05,
        )
    """

    weight: float = 0.1
    per_call_penalty: float = 0.01
    per_failure_penalty: float = 0.05
    max_penalty: float = 0.5
    penalize_tools: list[str] | None = None

    def __post_init__(self):
        self.name = "tool_penalty"

    def compute(self, rollout: RolloutResult) -> RewardComponent:
        """Compute penalty based on tool usage."""
        metrics = rollout.metrics
        total_calls = metrics.total_tool_calls
        failed_calls = metrics.failed_tool_calls

        # Filter by tool name if specified
        if self.penalize_tools:
            # Count only specific tools
            filtered_calls = 0
            for turn in rollout.turns:
                for tc in turn.tool_calls:
                    tool_name = tc.get("function", {}).get("name", "")
                    if tool_name in self.penalize_tools:
                        filtered_calls += 1
            total_calls = filtered_calls

        penalty = (
            total_calls * self.per_call_penalty
            + failed_calls * self.per_failure_penalty
        )
        penalty = min(penalty, self.max_penalty)

        # Return negative value (penalty)
        return self._make_component(
            value=-penalty,
            rationale=f"Tool calls: {total_calls}, failures: {failed_calls}",
            total_calls=total_calls,
            failed_calls=failed_calls,
        )


@dataclass
class LengthPenalty(BaseRewardFunction):
    """
    DAPO-style length penalty.

    Penalizes responses that exceed a target length, matching
    NeMo RL's reward shaping approach.

    Example:
        penalty = LengthPenalty(
            max_length=4096,
            buffer_length=512,
            penalty=0.5,
        )
    """

    weight: float = 0.1
    max_length: int = 4096
    buffer_length: int = 512
    penalty: float = 0.5

    def __post_init__(self):
        self.name = "length_penalty"

    def compute(self, rollout: RolloutResult) -> RewardComponent:
        """Compute length-based penalty."""
        total_tokens = rollout.metrics.total_generated_tokens
        expected_length = self.max_length - self.buffer_length

        if total_tokens <= expected_length:
            return self._make_component(
                value=0.0,
                rationale=f"Length {total_tokens} within limit {expected_length}",
                tokens=total_tokens,
            )

        # Calculate penalty
        exceed_length = total_tokens - expected_length
        penalty_value = min(
            (exceed_length / self.buffer_length) * self.penalty,
            self.penalty,
        )

        return self._make_component(
            value=-penalty_value,
            rationale=f"Length {total_tokens} exceeds limit by {exceed_length}",
            tokens=total_tokens,
            exceed_length=exceed_length,
        )


@dataclass
class TurnPenalty(BaseRewardFunction):
    """
    Penalty for excessive conversation turns.

    Encourages efficient task completion.

    Example:
        penalty = TurnPenalty(
            target_turns=5,
            per_extra_turn_penalty=0.02,
        )
    """

    weight: float = 0.1
    target_turns: int = 5
    per_extra_turn_penalty: float = 0.02
    max_penalty: float = 0.3

    def __post_init__(self):
        self.name = "turn_penalty"

    def compute(self, rollout: RolloutResult) -> RewardComponent:
        """Compute turn-based penalty."""
        total_turns = rollout.metrics.total_turns

        if total_turns <= self.target_turns:
            return self._make_component(
                value=0.0,
                rationale=f"Completed in {total_turns} turns (target: {self.target_turns})",
                turns=total_turns,
            )

        extra_turns = total_turns - self.target_turns
        penalty = min(extra_turns * self.per_extra_turn_penalty, self.max_penalty)

        return self._make_component(
            value=-penalty,
            rationale=f"Used {extra_turns} extra turns",
            turns=total_turns,
            extra_turns=extra_turns,
        )


@dataclass
class FormatReward(BaseRewardFunction):
    """
    Reward for following expected output format.

    Useful for structured outputs, code generation, etc.

    Example:
        reward = FormatReward(
            pattern=r"```python\\n.*\\n```",
            match_reward=0.2,
        )
    """

    weight: float = 0.2
    pattern: str | None = None
    required_strings: list[str] | None = None
    forbidden_strings: list[str] | None = None
    match_reward: float = 0.1
    violation_penalty: float = 0.1

    def __post_init__(self):
        self.name = "format_reward"
        if self.pattern:
            self._compiled_pattern = re.compile(self.pattern, re.DOTALL)
        else:
            self._compiled_pattern = None

    def compute(self, rollout: RolloutResult) -> RewardComponent:
        """Compute format-based reward/penalty."""
        # Get final assistant response
        final_response = ""
        for msg in reversed(rollout.message_log):
            if msg["role"] == "assistant":
                final_response = msg["content"]
                break

        score = 0.0
        reasons = []

        # Check pattern
        if self._compiled_pattern:
            if self._compiled_pattern.search(final_response):
                score += self.match_reward
                reasons.append("Pattern matched")
            else:
                score -= self.violation_penalty
                reasons.append("Pattern not matched")

        # Check required strings
        if self.required_strings:
            for req in self.required_strings:
                if req in final_response:
                    score += self.match_reward / len(self.required_strings)
                    reasons.append(f"Found: {req[:20]}...")
                else:
                    score -= self.violation_penalty / len(self.required_strings)
                    reasons.append(f"Missing: {req[:20]}...")

        # Check forbidden strings
        if self.forbidden_strings:
            for forbidden in self.forbidden_strings:
                if forbidden in final_response:
                    score -= self.violation_penalty
                    reasons.append(f"Contains forbidden: {forbidden[:20]}...")

        return self._make_component(
            value=score,
            rationale="; ".join(reasons) if reasons else "No format checks",
        )
