import typing as t

from dreadnode.core.agents.events import AgentEvent, AgentEventT
from dreadnode.core.agents.reactions import Reaction

EventCondition = t.Callable[[AgentEventT], bool]


class Hook(t.Generic[AgentEventT]):
    """Decorator class to mark a method as hook."""

    def __init__(
        self,
        func: t.Callable[..., t.Awaitable[Reaction | None]],
        event_type: type[AgentEventT],
        condition: EventCondition[AgentEventT] | None = None,
    ):
        self.func = func
        self.event_type = event_type
        self.condition = condition
        self.__name__ = func.__name__

    def __get__(
        self, obj: t.Any, objtype: type | None = None
    ) -> t.Callable[[AgentEvent], t.Awaitable[Reaction | None]]:
        if obj is None:
            return self

        async def bound_hook(event: AgentEvent) -> Reaction | None:
            if not isinstance(event, self.event_type):
                return None
            if self.condition is not None and not self.condition(event):
                return None
            return await self.func(obj, event)

        bound_hook.__name__ = self.__name__
        return bound_hook


def hook(
    event_type: type[AgentEventT],
    condition: EventCondition[AgentEventT] | None = None,
) -> t.Callable[[t.Callable[..., t.Awaitable[Reaction | None]]], Hook[AgentEventT]]:
    """
    Decorator to mark a method as hook.

    Args:
        event_type: The event type to listen for.
        condition: Optional filter condition.
    """

    def decorator(func: t.Callable[..., t.Awaitable[Reaction | None]]) -> Hook[AgentEventT]:
        return Hook(func, event_type, condition)

    return decorator
