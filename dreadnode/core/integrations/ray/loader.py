"""
Agent loader for Ray deployments.
"""

from __future__ import annotations

import importlib
import typing as t

from importlib.metadata import entry_points

if t.TYPE_CHECKING:
    from dreadnode.core.agents.agent import Agent


def discover_agents() -> dict[str, "Agent"]:
    """
    Discover all installed agent packages.

    Returns:
        Dictionary mapping agent names to Agent instances.

    Example:
        ```python
        agents = discover_agents()
        for name, agent in agents.items():
            print(f"Found agent: {name}")
        ```
    """
    agents: dict[str, Agent] = {}

    for ep in entry_points(group="dreadnode.agent"):
        try:
            module = ep.load()
            agent = _extract_agent_from_module(module, ep.name)
            if agent:
                agents[ep.name] = agent
        except Exception as e:
            print(f"Warning: Failed to load agent {ep.name}: {e}")

    return agents


def load_agent(name: str) -> "Agent":
    """
    Load a specific agent by name.

    Args:
        name: The entry point name of the agent (e.g., "org.my-agent").

    Returns:
        The loaded Agent instance.

    Raises:
        KeyError: If agent not found.

    Example:
        ```python
        agent = load_agent("acme.chat-agent")
        trajectory = await agent.run("Hello!")
        ```
    """
    for ep in entry_points(group="dreadnode.agent"):
        if ep.name == name:
            module = ep.load()
            agent = _extract_agent_from_module(module, name)
            if agent:
                return agent
            raise ValueError(f"No Agent found in module for {name}")

    raise KeyError(f"Agent not found: {name}")


def _extract_agent_from_module(module: t.Any, name: str) -> "Agent | None":
    """Extract an Agent instance from a loaded module."""
    from dreadnode.core.agents.agent import Agent

    # Check for common patterns:
    # 1. Module-level 'agent' variable
    if hasattr(module, "agent") and isinstance(module.agent, Agent):
        return module.agent

    # 2. Module-level 'Agent' variable
    if hasattr(module, "Agent") and isinstance(module.Agent, Agent):
        return module.Agent

    # 3. Function decorated with @agent that returns an Agent
    if hasattr(module, "agent") and callable(module.agent):
        result = module.agent
        if isinstance(result, Agent):
            return result

    # 4. 'run' or 'main' function that is an Agent
    for attr_name in ("run", "main", "default"):
        if hasattr(module, attr_name):
            attr = getattr(module, attr_name)
            if isinstance(attr, Agent):
                return attr

    # 5. Search all module attributes for an Agent instance
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        attr = getattr(module, attr_name)
        if isinstance(attr, Agent):
            return attr

    return None
