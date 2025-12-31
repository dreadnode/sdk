import typing as t

from dreadnode.core.environment import Environment


class SandboxEnvironment(Environment):
    """
        Minimal environment with custom lifecycle hooks.

        Example:
    ```python
            async def connect():
                client = await create_api_client()
                return {"client": client, "url": client.url}

            async def disconnect():
                await client.close()

            agent = Agent(
                name="api-agent",
                environment=SandboxEnvironment(
                    setup_fn=connect,
                    teardown_fn=disconnect,
                ),
                tools=[api_call, api_list],
            )
    ```
    """

    def __init__(
        self,
        *,
        setup_fn: t.Callable[[], t.Awaitable[dict[str, t.Any]]] | None = None,
        teardown_fn: t.Callable[[], t.Awaitable[None]] | None = None,
        reset_fn: t.Callable[[], t.Awaitable[dict[str, t.Any]]] | None = None,
        context: dict[str, t.Any] | None = None,
    ):
        self._setup_fn = setup_fn
        self._teardown_fn = teardown_fn
        self._reset_fn = reset_fn
        self._context = context or {}
        self._state: dict[str, t.Any] = {}

    async def setup(self) -> dict[str, t.Any]:
        if self._setup_fn:
            self._state = await self._setup_fn()
            return {**self._context, **self._state}
        return self._context

    async def teardown(self) -> None:
        if self._teardown_fn:
            await self._teardown_fn()
        self._state = {}

    async def reset(self) -> dict[str, t.Any]:
        if self._reset_fn:
            self._state = await self._reset_fn()
            return {**self._context, **self._state}
        return await super().reset()

    async def get_state(self) -> dict[str, t.Any]:
        return {**self._context, **self._state}
