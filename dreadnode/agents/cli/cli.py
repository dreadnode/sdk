"""Agent CLI for discovering and running agents."""

import typing as t

import rich

from dreadnode.cli.discoverable import DiscoverableCLI

if t.TYPE_CHECKING:
    from dreadnode.agents import Agent


async def _run_agent(agent: "Agent", input: str | None, raw: bool) -> None:
    """Run an agent with the given input."""
    async with agent.stream(input) as stream:
        async for event in stream:
            rich.print(event)


def _create_agent_cli() -> DiscoverableCLI["Agent"]:
    """Create the agent CLI using the shared discoverable pattern."""
    from dreadnode.agents import Agent
    from dreadnode.core.agents.format import format_agent, format_agents

    return DiscoverableCLI(
        name="agent",
        discovery_type=Agent,
        help_text="Discover and run agents.",
        object_name="agent",
        format_single=format_agent,
        format_multiple=format_agents,
        get_object_name=lambda a: a.name,
        get_object_description=lambda a: a.description,
        run_object=_run_agent,
        requires_input=True,
    )


# Create the CLI app
_cli = _create_agent_cli()
agent_cli = _cli.app
