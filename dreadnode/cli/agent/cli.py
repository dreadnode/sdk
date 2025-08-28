import contextlib
import itertools
import typing as t
from pathlib import Path

import cyclopts
import rich

from dreadnode.agent import Agent
from dreadnode.cli.agent.format import format_agent, format_agents_table
from dreadnode.discovery import DEFAULT_SEARCH_PATHS, discover
from dreadnode.meta import get_config_model, hydrate

cli = cyclopts.App("agent", help="Run and manage agents.", help_flags=[])


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
    discovered = discover(Agent, file)
    if not discovered:
        path_hint = file or ", ".join(DEFAULT_SEARCH_PATHS)
        rich.print(f"No agents found in {path_hint}")
        return

    grouped_by_path = itertools.groupby(discovered, key=lambda a: a.path)

    for path, discovered_agents in grouped_by_path:
        agents = [agent.obj for agent in discovered_agents]
        rich.print(f"Agents in [bold]{path}[/bold]:\n")
        if verbose:
            for agent in agents:
                rich.print(format_agent(agent))
        else:
            rich.print(format_agents_table(agents))


@cli.command(help_flags=[])
async def run(  # noqa: PLR0915
    agent: str,
    input: str,
    *tokens: t.Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    config: Path | None = None,
) -> None:
    """
    Run an agent by name, file, or module.

    - If just a file is passed, it will search for the first agent in that file ('my_agents.py').\n
    - If just an agent name is passed, it will search for that agent in the default files ('web_enum').\n
    - If the agent is specified with a file, it will run that specific agent in the given file ('my_agents.py:web_enum').\n
    - If the file is not specified, it defaults to searching for main.py, agent.py, or app.py.

    Args:
        agent: The agent to run, e.g., 'my_agents.py:basic' or 'basic'.
        input: The input to provide to the agent.
        config: Optional path to a TOML/YAML/JSON configuration file for the agent.
    """

    file_path: Path | None = None
    agent_name: str | None = None

    if agent is not None:
        agent_name = agent
        agent_as_path = Path(agent.split(":")[0]).with_suffix(".py")
        if agent_as_path.exists():
            file_path = agent_as_path
            agent_name = agent.split(":", 1)[-1] if ":" in agent else None

    path_hint = file_path or ", ".join(DEFAULT_SEARCH_PATHS)

    discovered = discover(Agent, file_path)
    if not discovered:
        rich.print(f":exclamation: No agents found in '{path_hint}'.")
        return

    agents_by_name = {d.name: d.obj for d in discovered}

    if agent_name is None:
        if len(discovered) > 1:
            rich.print(
                f"[yellow]Warning:[/yellow] Multiple agents found. Defaulting to the first one: '{next(iter(agents_by_name.keys()))}'."
            )
        agent_name = next(iter(agents_by_name.keys()))

    if agent_name not in agents_by_name:
        rich.print(f":exclamation: Agent '{agent_name}' not found in '{path_hint}'.")
        rich.print(f"Available agents are: {', '.join(agents_by_name.keys())}")
        return

    agent_blueprint = agents_by_name[agent_name]

    config_model = get_config_model(agent_blueprint)
    config_parameter = cyclopts.Parameter(name="*", group=f"Agent '{agent_name}' Config")(
        config_model
    )

    config_default = None
    with contextlib.suppress(Exception):
        config_default = config_model()
        config_parameter = t.Optional[config_parameter]  # type: ignore [assignment] # noqa: UP007

    async def agent_cli(*, config: t.Any = config_default) -> None:
        agent = hydrate(agent_blueprint, config)
        rich.print(f"Running agent: [bold]{agent.name}[/bold]")
        rich.print(agent)
        async with agent.stream(input) as stream:
            async for event in stream:
                rich.print(event)
                rich.print("---")

    agent_cli.__annotations__["config"] = config_parameter

    agent_app = cyclopts.App(help=f"Run the '{agent}' agent.", help_on_error=True)
    agent_app.default(agent_cli)

    if config:
        if not config.exists():
            rich.print(f":exclamation: Configuration file '{config}' does not exist.")
            return

        if config.suffix in {".toml"}:
            agent_app._config = cyclopts.config.Yaml(config, use_commands_as_keys=False)  # noqa: SLF001
        elif config.suffix in {".yaml", ".yml"}:
            agent_app._config = cyclopts.config.Toml(config, use_commands_as_keys=False)  # noqa: SLF001
        elif config.suffix in {".json"}:
            agent_app._config = cyclopts.config.Json(config, use_commands_as_keys=False)  # noqa: SLF001
        else:
            rich.print(f":exclamation: Unsupported configuration file format: '{config.suffix}'.")
            return

    # print(tokens)
    command, bound, unbound = agent_app.parse_args(tokens)
    # print(bound, unbound)
    await command(*bound.args, **bound.kwargs)
