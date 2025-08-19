from abc import ABC, abstractmethod
from collections.abc import Sequence

from pydantic import BaseModel, Field

from dreadnode.agent.events import Event, GenerationEnd, ToolEnd


class StopCondition(ABC, BaseModel):
    """
    A Pydantic-serializable condition that determines when an agent's run should stop.
    Conditions can be combined using & (AND) and | (OR).
    """

    @abstractmethod
    def __call__(self, events: Sequence[Event]) -> bool:
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

    def __call__(self, events: Sequence[Event]) -> bool:
        return all(cond(events) for cond in self.conditions)

    def __repr__(self) -> str:
        return f"({' & '.join(repr(cond) for cond in self.conditions)})"


class OrStopCondition(StopCondition):
    """Represents a logical OR of multiple conditions. Created via the | operator."""

    conditions: list[StopCondition]

    def __call__(self, events: Sequence[Event]) -> bool:
        return any(cond(events) for cond in self.conditions)

    def __repr__(self) -> str:
        return f"({' | '.join(repr(cond) for cond in self.conditions)})"


# --- Built-in, Concrete Conditions ---


class StopNever(StopCondition):
    """A condition that never stops the agent. Useful for forcing stalling behavior or specific tools for exit conditions."""

    def __call__(self, _: Sequence[Event]) -> bool:
        return False


class StopAfterSteps(StopCondition):
    """Terminates after a maximum number of LLM calls (steps)."""

    max_steps: int = Field(description="The maximum number of LLM generation steps to allow.")

    def __call__(self, events: Sequence[Event]) -> bool:
        step_count = sum(1 for event in events if isinstance(event, GenerationEnd))
        return step_count >= self.max_steps


class StopOnToolUse(StopCondition):
    """Terminates after a specific tool has been successfully used."""

    tool_name: str = Field(description="The name of the tool that should trigger termination.")

    def __call__(self, events: Sequence[Event]) -> bool:
        tool_events = [
            e for e in events if isinstance(e, ToolEnd) and e.tool_call.name == self.tool_name
        ]
        return any(event.tool_call.name == self.tool_name for event in tool_events)


class StopOnText(StopCondition):
    """Terminates if a specific string is mentioned in the last generated message."""

    text: str = Field(description="The text string to look for in the agent's final response.")
    case_sensitive: bool = Field(
        default=False, description="Whether the text match should be case-sensitive."
    )

    def __call__(self, events: Sequence[Event]) -> bool:
        if not events:
            return False

        if last_generation := next(
            (e for e in reversed(events) if isinstance(e, GenerationEnd)), None
        ):
            if self.case_sensitive:
                return self.text in last_generation.message.content
            return self.text.lower() in last_generation.message.content.lower()

        return False
