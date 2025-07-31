import typing as t
from pathlib import Path

import cyclopts
import rich

from dreadnode.agent.configurable import generate_config_model
from dreadnode.cli.agent.discover import discover_agents
from dreadnode.cli.agent.format import format_agent, format_agents_table

cli = cyclopts.App("agent", help="Run and manage agents.")


@cli.command(name=["list", "ls", "show"])
def show(
    file: Path | None = None,
    *,
    verbose: t.Annotated[
        bool,
        cyclopts.Parameter(
            ["--verbose", "-v"], help="Display detailed information for each agent."
        ),
    ] = False,
) -> None:
    """
    Discover and list available agents in a Python file.

    If no file is specified, searches for main.py, agent.py, or app.py.
    """
    discovery = discover_agents(file)
    if not discovery.agents:
        rich.print(f"No agents found in '[bold]{discovery.filepath}[/bold]'.")
        return

    rich.print(f"Agents in [bold]{discovery.filepath}[/bold]:\n")
    if verbose:
        for agent in discovery.agents.values():
            rich.print(format_agent(agent))
    else:
        rich.print(format_agents_table(list(discovery.agents.values())))


@cli.command()
async def run(
    agent: t.Annotated[
        str,
        cyclopts.Parameter(
            help="The agent to run, e.g., 'my_agents.py:basic' or 'my_agents:basic'."
        ),
    ],
    **args: t.Any,
) -> None:
    """
    Run an agent with dynamic configuration. (Not yet implemented)
    """

    file_path: Path | None = None
    agent_name: str | None = None

    if agent is not None:
        if ":" not in agent:
            file_str, agent_name = agent, None
        else:
            file_str, agent_name = agent.split(":", 1)

        if not file_str.endswith(".py"):
            file_str += ".py"

        file_path = Path(file_str)

    discovered = discover_agents(file_path)
    if not discovered.agents:
        rich.print(f":exclamation: No agents found in '{file_path}'.")
        return

    if agent_name is None:
        if len(discovered.agents) > 1:
            rich.print(
                f"[yellow]Warning:[/yellow] Multiple agents found. Defaulting to the first one: '{next(iter(discovered.agents.keys()))}'."
            )
        agent_name = next(iter(discovered.agents.keys()))

    if agent_name not in discovered.agents:
        rich.print(f":exclamation: Agent '{agent_name}' not found in '{file_path}'.")
        rich.print(f"Available agents are: {', '.join(discovered.agents.keys())}")
        return

    agent_blueprint = discovered.agents[agent_name]

    config_model = generate_config_model(agent_blueprint)
    config_parameter = cyclopts.Parameter(name="*")(config_model)

    async def agent_run(config: t.Any) -> None:
        print(config)

    agent_run.__annotations__["config"] = config_parameter

    agent_cli = cyclopts.App(help=f"Run the '{agent}' agent.", help_on_error=True)
    agent_cli.default(agent_run)

    print(args)
    command, bound = agent_cli.parse_args(list(args))
    await command(*bound.args, **bound.kwargs)
