import typing as t
from typing import Protocol

if t.TYPE_CHECKING:
    from dreadnode.eval.events import EvalEvent
    from dreadnode.eval.reactions import EvalReaction


@t.runtime_checkable
class EvalHook(Protocol):
    """Protocol for evaluation lifecycle hooks."""

    async def __call__(
        self,
        event: "EvalEvent",
    ) -> "EvalReaction | None":
        """
        Process an evaluation event and optionally return a reaction.

        Args:
            event: The evaluation event to process

        Returns:
            An optional reaction that modifies evaluation behavior
        """
        ...
