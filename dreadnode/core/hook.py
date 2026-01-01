import typing as t

from dreadnode.core.agents.events import AgentEvent, AgentEventT
from dreadnode.core.agents.reactions import Reaction

EventCondition = t.Callable[[AgentEventT], bool]


class Hook(t.Generic[AgentEventT]):
    """
    Decorator class to mark a function or method as a hook.

    Can be used in three ways:

    1. Standalone async function (direct hook):
        @hook(GenerationStep)
        async def my_hook(event: GenerationStep) -> Reaction | None:
            ...

    2. Factory function (returns hook):
        @hook(AgentError)
        def backoff_on_error(max_tries: int = 5) -> Hook:
            async def inner(event: AgentError) -> Reaction | None:
                ...
            return inner

    3. Class method:
        class MyHooks:
            @hook(GenerationStep)
            async def my_hook(self, event: GenerationStep) -> Reaction | None:
                ...
    """

    def __init__(
        self,
        func: t.Callable[..., t.Any],
        event_type: type[AgentEventT],
        condition: EventCondition[AgentEventT] | None = None,
    ):
        self.func = func
        self.event_type = event_type
        self.condition = condition
        self.__name__ = func.__name__

    async def __call__(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        """
        Call the hook.

        For direct async hooks: passes the event and filters by type/condition.
        For factory functions: passes args to create the inner hook.
        """
        import inspect

        # Check if this is being called with an event (direct hook pattern)
        if args and isinstance(args[0], AgentEvent):
            event = args[0]
            if not isinstance(event, self.event_type):
                return None
            if self.condition is not None:
                condition_result = self.condition(event)
                if inspect.isawaitable(condition_result):
                    condition_result = await condition_result
                if not condition_result:
                    return None
            result = self.func(event)
            if inspect.isawaitable(result):
                return await result
            return result

        # Otherwise, this is a factory function being called with config args
        return self.func(*args, **kwargs)

    def __get__(
        self, obj: t.Any, objtype: type | None = None
    ) -> t.Callable[[AgentEvent], t.Awaitable[Reaction | None]]:
        """Descriptor protocol for method hooks."""
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
