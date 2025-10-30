import contextlib
import inspect
import itertools
import typing as t
from inspect import isawaitable
from pathlib import Path

import cyclopts
import rich

from dreadnode.cli.shared import DreadnodeConfig
from dreadnode.discovery import DEFAULT_SEARCH_PATHS, discover
from dreadnode.logging_ import console as logging_console
from dreadnode.meta import get_config_model, hydrate
from dreadnode.meta.introspect import flatten_model

cli = cyclopts.App("agent", help="Discover and run agents.")


@cli.command(name=["list", "ls", "show"])
def show(
    file: Path | None = None,
    *,
    verbose: t.Annotated[
        bool,
        cyclopts.Parameter(["--verbose", "-v"], help="Display detailed information."),
    ] = False,
) -> None:
    """
    Discover and list available agents in a Python file.

    If no file is specified, searches in standard paths.
    """
    from dreadnode.agent import Agent
    from dreadnode.agent.format import format_agent, format_agents

    discovered = discover(Agent, file)
    if not discovered:
        path_hint = file or ", ".join(DEFAULT_SEARCH_PATHS)
        rich.print(f"No agents found in {path_hint}.")
        return

    grouped_by_path = itertools.groupby(discovered, key=lambda a: a.path)
    for path, discovered_agents in grouped_by_path:
        agents = [agent.obj for agent in discovered_agents]
        rich.print(f"Agents in [bold]{path}[/bold]:\n")
        if verbose:
            for agent in agents:
                rich.print(format_agent(agent))
        else:
            rich.print(format_agents(agents))


@cli.command()
async def run(  # noqa: PLR0912, PLR0915
    agent: str,
    *tokens: t.Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    config: Path | None = None,
    raw: t.Annotated[bool, cyclopts.Parameter(["-r", "--raw"], negative=False)] = False,
    dn_config: DreadnodeConfig | None = None,
) -> None:
    """
    Run an agent by name, file, or module.

    - If just a file is passed, it will search for the first agent in that file ('my_agents.py').\n
    - If just an agent name is passed, it will search for that agent in the default files ('web_enum').\n
    - If the agent is specified with a file, it will run that specific agent in the given file ('my_agents.py:web_enum').\n
    - If the file is not specified, it defaults to searching for main.py, agent.py, or app.py.

    **To get detailed help for a specific agent, use `dreadnode agent run <agent> help`.**

    Args:
        agent: The agent to run, e.g., 'my_agents.py:basic' or 'basic'.
        config: Optional path to a TOML/YAML/JSON configuration file for the agent.
        raw: If set, only display raw logging output without additional formatting.
    """
    from dreadnode.agent import Agent

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
        rich.print(f":exclamation: No agents found in {path_hint}.")
        return

    agents_by_name = {d.obj.name: d.obj for d in discovered}
    agents_by_lower_name = {k.lower(): v for k, v in agents_by_name.items()}
    if agent_name is None:
        if len(discovered) > 1:
            rich.print(
                f"[yellow]Warning:[/yellow] Multiple agents found. Defaulting to the first one: '{next(iter(agents_by_name.keys()))}'."
            )
        agent_name = next(iter(agents_by_name.keys()))

    if agent_name.lower() not in agents_by_lower_name:
        rich.print(f":exclamation: Agent '{agent_name}' not found in {path_hint}.")
        rich.print(f"Available agents are: {', '.join(agents_by_name.keys())}")
        return

    agent_blueprint = agents_by_lower_name[agent_name.lower()]

    config_model = get_config_model(agent_blueprint)
    config_annotation = cyclopts.Parameter(name="*", group="Agent Config")(config_model)
    config_default: t.Any = inspect.Parameter.empty
    with contextlib.suppress(Exception):
        config_default = config_model()

    async def agent_cli(
        input: t.Annotated[str, cyclopts.Parameter(help="Input to the agent")],
        *,
        config: t.Any = config_default,
        dn_config: DreadnodeConfig | None = dn_config,
    ) -> None:
        dn_config = dn_config or DreadnodeConfig()
        if raw and dn_config.log_level is None:
            dn_config.log_level = "info"
        dn_config.apply()
        agent = hydrate(agent_blueprint, config)

        rich.print(f"Running agent: [bold]{agent.name}[/bold] with config:")
        for key, value in flatten_model(config).items():
            rich.print(f" |- {key}: {value}")
        rich.print()

        async with agent.stream(input) as stream:
            async for event in stream:
                rich.print(event)

    agent_cli.__annotations__["config"] = config_annotation

    help_text = f"Run the '{agent_name}' agent."
    if agent_blueprint.description:
        help_text += "\n\n" + agent_blueprint.description

    agent_app = cyclopts.App(
        name=agent_name,
        help=help_text,
        help_on_error=True,
        help_flags=("help"),
        version_flags=(),
        console=logging_console,
    )
    agent_app.default(agent_cli)

    if config:
        if not config.exists():
            rich.print(f":exclamation: Configuration file '{config}' does not exist.")
            return

        if config.suffix in {".toml"}:
            agent_app._config = cyclopts.config.Toml(config, use_commands_as_keys=False)  # type: ignore[assignment] # noqa: SLF001
        elif config.suffix in {".yaml", ".yml"}:
            agent_app._config = cyclopts.config.Yaml(config, use_commands_as_keys=False)  # type: ignore[assignment] # noqa: SLF001
        elif config.suffix in {".json"}:
            agent_app._config = cyclopts.config.Json(config, use_commands_as_keys=False)  # type: ignore[assignment] # noqa: SLF001
        else:
            rich.print(f":exclamation: Unsupported configuration file format: '{config.suffix}'.")
            return

    command, bound, _ = agent_app.parse_args(tokens)

    result = command(*bound.args, **bound.kwargs)
    if isawaitable(result):
        await result
