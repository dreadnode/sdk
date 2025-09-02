import contextlib
import itertools
import typing as t
from inspect import isawaitable
from pathlib import Path

import cyclopts
import rich

from dreadnode.agent.format import format_tool, format_tools_table
from dreadnode.agent.tools import Toolset
from dreadnode.discovery import DEFAULT_TOOL_SEARCH_PATH, discover
from dreadnode.meta import get_config_model, hydrate
from dreadnode.meta.introspect import flatten_model

cli = cyclopts.App("tools", help="Run and manage tools.")


@cli.command(name=["list", "ls", "show"])
def show(
    file: Path | None = None,
    *,
    verbose: t.Annotated[
        bool,
        cyclopts.Parameter(["--verbose", "-v"], help="Display detailed information for each tool."),
    ] = False,
) -> None:
    """
    Discover and list available tools in a Python file.

    If no file is specified, searches for `tool.py`.
    """
    if not file:
        file = DEFAULT_TOOL_SEARCH_PATH
    discovered = discover(Toolset, file)
    if not discovered:
        path_hint = file or ", ".join(str(DEFAULT_TOOL_SEARCH_PATH))
        rich.print(f"No tools found in {path_hint}")
        return

    grouped_by_path = itertools.groupby(discovered, key=lambda a: a.path)

    for path, discovered_tools in grouped_by_path:
        tools = [tool.obj for tool in discovered_tools]
        rich.print(f"Tools in [bold]{path}[/bold]:\n")
        if verbose:
            for tool in tools:
                rich.print(format_tool(tool))
        else:
            rich.print(format_tools_table(tools))


@cli.command()
async def install(
    tool: str,
    *,
    server: t.Annotated[
        str | None,
        cyclopts.Parameter(name=["--server", "-s"], help="URL of the server to clone from."),
    ] = None,
    profile: t.Annotated[
        str | None,
        cyclopts.Parameter(
            name=["--profile", "-p"], help="Profile alias to use for authentication."
        ),
    ] = None,
    dest: t.Annotated[
        Path | None,
        cyclopts.Parameter(
            name=["--dest", "-d"],
            help="Destination directory to install the tool into. Defaults to ~/.dreadnode/tools/<tool>.",
        ),
    ] = None,
) -> None:
    """
    Install a tool from a GitHub repository.

    The tool should be in a repository under the `dreadnode-tools` organization.
    For example, to install the `web_enum` tool, you would run:

        dreadnode tools install web_enum

    This would clone from:

    """


@cli.command()
async def run(  # noqa: PLR0912, PLR0915
    tool: str,
    *tokens: t.Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    config: Path | None = None,
) -> None:
    """
    Run an tool by name, file, or module.

    - If just a file is passed, it will search for the first tool in that file ('my_tools.py').\n
    - If just an tool name is passed, it will search for that tool in the default files ('web_enum').\n
    - If the tool is specified with a file, it will run that specific tool in the given file ('my_tools.py:web_enum').\n
    - If the file is not specified, it defaults to searching for main.py, tool.py, or app.py.

    **To get detailed help for a specific tool, use `dreadnode tool run <tool> help`.**

    Args:
        tool: The tool to run, e.g., 'my_tools.py:basic' or 'basic'.
        config: Optional path to a TOML/YAML/JSON configuration file for the tool.
    """

    file_path: Path | None = None
    tool_name: str | None = None

    if tool is not None:
        tool_name = tool
        tool_as_path = Path(tool.split(":")[0]).with_suffix(".py")
        if tool_as_path.exists():
            file_path = tool_as_path
            tool_name = tool.split(":", 1)[-1] if ":" in tool else None

    path_hint = file_path or ", ".join(str(DEFAULT_TOOL_SEARCH_PATH))

    discovered = discover(Toolset, file_path)
    if not discovered:
        rich.print(f":exclamation: No tools found in '{path_hint}'.")
        return

    tools_by_name = {d.name: d.obj for d in discovered}

    if tool_name is None:
        if len(discovered) > 1:
            rich.print(
                f"[yellow]Warning:[/yellow] Multiple tools found. Defaulting to the first one: '{next(iter(tools_by_name.keys()))}'."
            )
        tool_name = next(iter(tools_by_name.keys()))

    if tool_name not in tools_by_name:
        rich.print(f":exclamation: Toolset '{tool_name}' not found in '{path_hint}'.")
        rich.print(f"Available tools are: {', '.join(tools_by_name.keys())}")
        return

    tool_blueprint = tools_by_name[tool_name]

    config_model = get_config_model(tool_blueprint)
    config_parameter = cyclopts.Parameter(name="*", group="Tool Config")(config_model)

    config_default = None
    with contextlib.suppress(Exception):
        config_default = config_model()
        config_parameter = config_parameter | None  # type: ignore [assignment]

    async def tool_cli(
        config: t.Any = config_default,
    ) -> None:
        flat_config = {k: v for k, v in flatten_model(config).items() if v is not None}
        tool = hydrate(tool_blueprint, config)

        rich.print(f"Running tool: [bold]{tool.name}[/bold] with config:")
        for key, value in flat_config.items():
            rich.print(f" |- {key}: {value}")
        rich.print()

        rich.print("[bold]Tool Output: TODO[/bold]\n")

        # with run_span(name_prefix=f"tool-{tool.name}", params=flat_config, tags=tool.variant):
        #     log_input("user_input", input)
        #     async with tool.stream(input) as stream:
        #         async for event in stream:
        #             rich.print(event)

    tool_cli.__annotations__["config"] = config_parameter

    tool_app = cyclopts.App(
        name=tool_name,
        help=f"Run the '{tool_name}' tool.",
        help_on_error=True,
        help_flags=("help"),
        version_flags=(),
    )
    tool_app.default(tool_cli)

    if config:
        if not config.exists():
            rich.print(f":exclamation: Configuration file '{config}' does not exist.")
            return

        if config.suffix in {".toml"}:
            tool_app._config = cyclopts.config.Toml(config, use_commands_as_keys=False)  # noqa: SLF001
        elif config.suffix in {".yaml", ".yml"}:
            tool_app._config = cyclopts.config.Yaml(config, use_commands_as_keys=False)  # noqa: SLF001
        elif config.suffix in {".json"}:
            tool_app._config = cyclopts.config.Json(config, use_commands_as_keys=False)  # noqa: SLF001
        else:
            rich.print(f":exclamation: Unsupported configuration file format: '{config.suffix}'.")
            return

    command, bound, _ = tool_app.parse_args(tokens)

    result = command(*bound.args, **bound.kwargs)
    if isawaitable(result):
        await result
