"""Agent package loader for published agent packages."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from dreadnode.core.packaging.loader import BaseLoader
from dreadnode.core.packaging.manifest import AgentManifest

if TYPE_CHECKING:
    from dreadnode.core.agents import Agent


class AgentPackage(BaseLoader):
    """Loader for published agent packages.

    This class loads agents that have been published as DN packages
    with entry points in the 'dreadnode.agents' group.

    Example:
        >>> from dreadnode.agents import AgentPackage
        >>>
        >>> # Load a published agent
        >>> pkg = AgentPackage("my-org-classifier")
        >>> print(pkg.entrypoint)  # 'main:run'
        >>>
        >>> # Get the Agent instance
        >>> agent = pkg.load()
        >>> result = await agent.run("Classify this text")
    """

    entry_point_group = "dreadnode.agents"
    manifest_class = AgentManifest

    @property
    def entrypoint(self) -> str:
        """Agent entrypoint (e.g., 'main:run')."""
        return self.manifest.entrypoint

    @property
    def toolsets(self) -> list[str]:
        """List of referenced toolsets."""
        return self.manifest.toolsets

    @property
    def models(self) -> list[str]:
        """List of referenced models."""
        return self.manifest.models

    @property
    def datasets(self) -> list[str]:
        """List of referenced datasets."""
        return self.manifest.datasets

    def load(self, **kwargs: Any) -> Agent:
        """Load and return the Agent instance.

        Args:
            **kwargs: Override arguments for the agent.

        Returns:
            Agent instance ready for execution.

        Example:
            >>> pkg = AgentPackage("my-agent")
            >>> agent = pkg.load()
            >>> result = await agent.run("Do something")
        """
        # Parse entrypoint (format: "module:attribute")
        module_name, attr_name = self.entrypoint.rsplit(":", 1)

        # The module is already loaded via entry point
        # Navigate to the correct attribute
        module = self._module
        for part in module_name.split("."):
            if part and hasattr(module, part):
                module = getattr(module, part)

        agent = getattr(module, attr_name)

        # If callable, call it to get the agent
        if callable(agent) and not hasattr(agent, "run"):
            agent = agent()

        # Apply any overrides
        if kwargs:
            agent = agent.with_(**kwargs)

        return agent


def load_agent(name: str, **kwargs: Any) -> Agent:
    """Load a published agent by name.

    Args:
        name: Agent package name (entry point name).
        **kwargs: Override arguments for the agent.

    Returns:
        Agent instance.

    Example:
        >>> agent = load_agent("my-classifier")
        >>> result = await agent.run("Classify this")
    """
    pkg = AgentPackage(name)
    return pkg.load(**kwargs)
