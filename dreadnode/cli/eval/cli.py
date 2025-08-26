import contextlib
import typing as t
from pathlib import Path

import cyclopts
import rich

from dreadnode.configurable import generate_config_type_for_agent, hydrate_agent
from dreadnode.discovery import discover
from dreadnode.eval.eval import Eval

cli = cyclopts.App("eval", help="Run and manage evaluations.", help_flags=[])


@cli.command(name=["list", "ls", "show"])
def show(
    file: Path | None = None,
    *,
    verbose: t.Annotated[bool, cyclopts.Parameter(["--verbose", "-v"])] = False,
) -> None:
    """
    Discover and list available evals.

    If no file is specified, searches in main.py, agent.py, eval.py, and app.py.

    Args:
        file: Optional path to a specific file to search for evals.
        verbose: If true, shows detailed information about each eval.
    """

    discovered = discover(Eval, file)
    if not discovered:
        rich.print("No evals found.")
        return

    for filepath, objects in discovered.items():
        if not objects:
            continue

        rich.print(f"Evals in [bold]{filepath}[/bold]:\n")
        if verbose:
            for obj in objects:
                rich.print(format_verbose(obj))
        else:
            rich.print(format_table(objects))


@cli.command
async def run(
    eval: str,
    *tokens: t.Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    config: Path | None = None,
) -> None:
    """
    Run an eval by name, file, or module.

    - If just a file is passed, it will search for the first eval in that file ('my_evals.py').\n
    - If just an eval name is passed, it will search for that eval in default files ('web_enum').\n
    - If the eval is specified with a file, it will run that specific eval in the given file ('my_evals.py:web_enum').\n
    - If the file is not specified, it defaults to searching for main.py, eval.py, or app.py.

    Args:
        eval: The eval to run, e.g., 'my_evals.py:comprehensive' or 'comprehensive'.
        input: The input to provide to the eval.
        config: Optional path to a TOML/YAML/JSON configuration file for the eval.
    """

    file_path_str = eval.split(":", 1)[0]
    name = eval.split(":", 1)[-1] if ":" in eval else None

    evals = discover(Eval, Path(file_path_str))
    if not evals:
        rich.print(f":exclamation: No evals found in '{file_path_str}'.")
        return

    if name is None:
        if len(evals) > 1:
            rich.print(
                f"[yellow]Warning:[/yellow] Multiple evalss found. Defaulting to '{evals[0].name}'."
            )
        blueprint = evals[0].obj
    else:
        try:
            blueprint = next(d.obj for d in discoveries if d.name == name)
        except StopIteration:
            available = ", ".join(d.name for d in discoveries)
            rich.print(f":exclamation: {noun.capitalize()} '{name}' not found.")
            rich.print(f"Available: {available}")
            return

    agent_blueprint = discovered.agents[agent_name]

    config_model = generate_config_type_for_agent(agent_blueprint)
    config_parameter = cyclopts.Parameter(name="*", group=f"Agent '{agent_name}' Config")(
        config_model
    )

    config_default = None
    with contextlib.suppress(Exception):
        config_default = config_model()
        config_parameter = t.Optional[config_parameter]  # type: ignore [assignment] # noqa: UP007

    async def agent_cli(*, config: t.Any = config_default) -> None:
        agent = hydrate_agent(agent_blueprint, config)
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

    command, bound, _ = agent_app.parse_args(tokens)
    await command(*bound.args, **bound.kwargs)
