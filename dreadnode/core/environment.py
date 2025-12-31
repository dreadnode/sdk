import typing as t
from abc import ABC, abstractmethod


class Environment(ABC):
    """
    Base class for agent execution environments.

    An environment manages the lifecycle of an external system
    (container, kernel, API connection, etc.) and provides context
    to the agent.
    """

    @abstractmethod
    async def setup(self) -> dict[str, t.Any]:
        """
        Initialize the environment.

        Returns:
            Context dict passed to the agent (injected into instructions).
        """
        ...

    @abstractmethod
    async def teardown(self) -> None:
        """Clean up the environment."""
        ...

    async def reset(self) -> dict[str, t.Any]:
        """Reset for a new attempt. Default: teardown + setup."""
        await self.teardown()
        return await self.setup()

    async def get_state(self) -> dict[str, t.Any]:
        """Get current state (for debugging/logging)."""
        return {}

    async def __aenter__(self) -> "Environment":
        await self.setup()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.teardown()
