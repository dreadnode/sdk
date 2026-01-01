"""
Types for rollout orchestration.

Matches NeMo RL's message_log structure while supporting DN SDK integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, NotRequired, TypedDict

import torch


class MessageRole(str, Enum):
    """Message roles in conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    ENVIRONMENT = "environment"


class Message(TypedDict):
    """
    Single message in conversation log.

    Matches NeMo RL's LLMMessageLogType structure.
    """

    role: str
    content: str
    # Optional fields for training
    token_ids: NotRequired[torch.Tensor]
    generation_logprobs: NotRequired[torch.Tensor]
    # Tool call specific
    tool_calls: NotRequired[list[dict[str, Any]]]
    tool_call_id: NotRequired[str]
    name: NotRequired[str]  # Tool name for tool responses


# Type alias matching NeMo RL
MessageLog = list[Message]


class RolloutConfig(TypedDict):
    """Configuration for rollout execution."""

    # Turn limits
    max_turns: int  # Maximum conversation turns
    max_tokens_per_turn: NotRequired[int]  # Max tokens per generation
    max_total_tokens: NotRequired[int]  # Max tokens across all turns

    # Termination
    stop_strings: NotRequired[list[str]]  # Generation stop strings
    terminate_on_tool_error: NotRequired[bool]  # Stop if tool fails

    # Chat template
    chat_template: NotRequired[str]  # Template name or path
    tool_parser: NotRequired[str]  # Tool call parser (hermes, llama3, mistral)

    # Tokenizer (for token counting)
    tokenizer_name: NotRequired[str]

    # Execution
    timeout_per_turn: NotRequired[float]  # Seconds per turn
    timeout_total: NotRequired[float]  # Total rollout timeout


@dataclass
class TurnResult:
    """Result from a single turn of interaction."""

    turn_number: int

    # Generation
    generated_text: str
    generated_tokens: int
    input_tokens: int
    logprobs: torch.Tensor | None = None

    # Environment response
    env_observation: str | None = None
    env_tokens: int = 0
    reward: float = 0.0

    # Status
    terminated: bool = False
    truncated: bool = False
    error: str | None = None

    # Timing
    generation_time: float = 0.0
    env_time: float = 0.0

    # Tool calls made this turn
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RolloutMetrics:
    """Aggregated metrics from a rollout."""

    # Turns
    total_turns: int = 0
    completed_turns: int = 0

    # Tokens
    total_input_tokens: int = 0
    total_generated_tokens: int = 0
    total_env_tokens: int = 0
    mean_tokens_per_turn: float = 0.0

    # Rewards
    total_reward: float = 0.0
    mean_reward_per_turn: float = 0.0
    final_reward: float = 0.0

    # Termination
    natural_termination: bool = False  # Env signaled done
    truncated: bool = False  # Hit max turns/tokens
    errored: bool = False

    # Timing
    total_time: float = 0.0
    mean_generation_time: float = 0.0
    mean_env_time: float = 0.0

    # Tool usage
    total_tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0


@dataclass
class RolloutResult:
    """
    Complete result from a rollout.

    Contains the full message log, per-turn results, and aggregated metrics.
    """

    # Identification
    rollout_id: str
    agent_id: str | None = None
    goal: str = ""

    # Message history (NeMo RL compatible)
    message_log: MessageLog = field(default_factory=list)

    # Per-turn breakdown
    turns: list[TurnResult] = field(default_factory=list)

    # Aggregated metrics
    metrics: RolloutMetrics = field(default_factory=RolloutMetrics)

    # Metadata
    config: RolloutConfig | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # For training
    success: bool = False
    final_reward: float = 0.0

    # Additional context
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_trajectory_record(self) -> dict[str, Any]:
        """Convert to trajectory JSONL record for training."""
        return {
            "id": self.rollout_id,
            "agent_id": self.agent_id,
            "goal": self.goal,
            "messages": [
                {"role": m["role"], "content": m["content"]} for m in self.message_log
            ],
            "success": self.success,
            "reward": self.final_reward,
            "metrics": {
                "turns": self.metrics.total_turns,
                "tokens": self.metrics.total_generated_tokens,
                "tool_calls": self.metrics.total_tool_calls,
            },
            "metadata": self.metadata,
        }

    def to_openai_format(self) -> list[dict[str, Any]]:
        """Convert message log to OpenAI format for training."""
        messages = []
        for msg in self.message_log:
            formatted: dict[str, Any] = {"role": msg["role"], "content": msg["content"]}
            if "tool_calls" in msg and msg["tool_calls"]:
                formatted["tool_calls"] = msg["tool_calls"]
            if "tool_call_id" in msg:
                formatted["tool_call_id"] = msg["tool_call_id"]
            if "name" in msg:
                formatted["name"] = msg["name"]
            messages.append(formatted)
        return messages

    def to_dn_trajectory(self) -> "Trajectory":
        """
        Convert to DN SDK Trajectory.

        Returns:
            dreadnode.core.agents.trajectory.Trajectory
        """
        from dreadnode.core.agents.trajectory import trajectory_from_openai_format

        return trajectory_from_openai_format(self.to_openai_format())

    @classmethod
    def from_dn_trajectory(
        cls,
        trajectory: "Trajectory",
        goal: str = "",
        success: bool = False,
        reward: float = 0.0,
    ) -> "RolloutResult":
        """
        Create RolloutResult from DN SDK Trajectory.

        Args:
            trajectory: DN SDK Trajectory object
            goal: The goal/prompt for this rollout
            success: Whether the task was successful
            reward: Final reward value

        Returns:
            RolloutResult instance
        """
        from dreadnode.core.agents.trajectory import trajectory_to_openai_format

        messages = trajectory_to_openai_format(trajectory)
        message_log: MessageLog = [
            Message(role=m["role"], content=m.get("content", "")) for m in messages
        ]

        # Build metrics from trajectory
        metrics = RolloutMetrics(
            total_turns=len(trajectory.steps),
            completed_turns=len(trajectory.steps),
            total_input_tokens=trajectory.usage.input_tokens,
            total_generated_tokens=trajectory.usage.output_tokens,
            final_reward=reward,
        )

        return cls(
            rollout_id=str(trajectory.session_id),
            agent_id=str(trajectory.agent_id) if trajectory.agent_id else None,
            goal=goal,
            message_log=message_log,
            metrics=metrics,
            success=success,
            final_reward=reward,
        )


# Type alias for Trajectory (avoid circular import)
if False:  # TYPE_CHECKING
    from dreadnode.core.agents.trajectory import Trajectory
