from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel, Field

from dreadnode.agent.events import AgentEvent, GenerationEnd, ToolCallEnd


class StopCondition(ABC, BaseModel):
    """
    A Pydantic-serializable condition that determines when an agent's run should stop.
    Conditions can be combined using & (AND) and | (OR).
    """

    @abstractmethod
    def __call__(self, events: Sequence[AgentEvent]) -> bool:
        """
        Checks if the termination condition has been met against the history of a run.

        Args:
            events: A sequence of all events that have occurred in the current run.

        Returns:
            True if the run should terminate, False otherwise.
        """

    def __and__(self, other: "StopCondition") -> "AndStopCondition":
        """Combines this condition with another using AND logic."""
        return AndStopCondition(conditions=[self, other])

    def __or__(self, other: "StopCondition") -> "OrStopCondition":
        """Combines this condition with another using OR logic."""
        return OrStopCondition(conditions=[self, other])


class AndStopCondition(StopCondition):
    """Represents a logical AND of multiple conditions. Created via the & operator."""

    conditions: list[StopCondition]

    def __call__(self, events: Sequence[AgentEvent]) -> bool:
        return all(cond(events) for cond in self.conditions)


class OrStopCondition(StopCondition):
    """Represents a logical OR of multiple conditions. Created via the | operator."""

    conditions: list[StopCondition]

    def __call__(self, events: Sequence[AgentEvent]) -> bool:
        return any(cond(events) for cond in self.conditions)


# --- Built-in, Concrete Conditions ---


class StopAfterSteps(StopCondition):
    """Terminates after a maximum number of LLM calls (steps)."""

    max_steps: int = Field(description="The maximum number of LLM generation steps to allow.")

    def __call__(self, events: Sequence[AgentEvent]) -> bool:
        step_count = sum(1 for event in events if isinstance(event, GenerationEnd))
        return step_count >= self.max_steps


class StopOnToolUse(StopCondition):
    """Terminates after a specific tool has been successfully used."""

    tool_name: str = Field(description="The name of the tool that should trigger termination.")

    def __call__(self, events: Sequence[AgentEvent]) -> bool:
        if not events:
            return False
        last_event = events[-1]
        return isinstance(last_event, ToolCallEnd) and last_event.tool_call.name == self.tool_name


class StopOnText(StopCondition):
    """Terminates if a specific string is mentioned in the last generated message."""

    text: str = Field(description="The text string to look for in the agent's final response.")
    case_sensitive: bool = Field(
        default=False, description="Whether the text match should be case-sensitive."
    )

    def __call__(self, events: Sequence[AgentEvent]) -> bool:
        if not events:
            return False

        if last_generation := next(
            (e for e in reversed(events) if isinstance(e, GenerationEnd)), None
        ):
            if self.case_sensitive:
                return self.text in last_generation.generated_message.content
            return self.text.lower() in last_generation.generated_message.content.lower()

        return False
