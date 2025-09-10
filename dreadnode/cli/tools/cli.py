import contextlib
import pathlib
import shutil
import typing as t
from inspect import isawaitable

import cyclopts
import rich
from rich.prompt import Prompt

from dreadnode.agent.tools import Toolset
from dreadnode.cli.api import create_api_client
from dreadnode.cli.github import (
    GithubRepo,
    download_and_unzip_archive,
    validate_server_for_clone,
)
from dreadnode.cli.tools.discover import discover_tools
from dreadnode.constants import DEFAULT_TOOL_SEARCH_PATH
from dreadnode.discovery import discover
from dreadnode.meta import get_config_model, hydrate
from dreadnode.meta.introspect import flatten_model
from dreadnode.user_config import UserConfig

cli = cyclopts.App("tools", help="Run and manage tools.")


@cli.command(name=["list"])
def show(
    file: pathlib.Path | None = DEFAULT_TOOL_SEARCH_PATH,
    *,
    verbose: t.Annotated[
        bool,
        cyclopts.Parameter(["--verbose", "-v"], help="Display detailed information for each tool."),
    ] = False,
) -> None:
    """
    Discover and list available agents in a Python file.

    If no file is specified, searches for `tool.py`.
    """
    tools = discover_tools(file)

    if not tools:
        rich.print(f":exclamation: No tools found in '{file}'.")
        return

    rich.print(f":mag: Found {len(tools)} tool(s) in '{file}':\n")

    for tool in tools:
        rich.print(f" - [bold]{tool}[/bold]")
        if verbose:
            rich.print("   - Description: TODO")
            rich.print("   - Version: TODO")
            rich.print("   - Author: TODO")
            rich.print()


@cli.command()
def clone(
    repo: t.Annotated[str, cyclopts.Parameter(help="Repository name or URL")] = "dreadnode/tools",
    tool_path: t.Annotated[
        pathlib.Path | None,
        cyclopts.Parameter(help="The target directory"),
    ] = DEFAULT_TOOL_SEARCH_PATH,
) -> None:
    """Clone the tool repository to a local directory"""

    github_repo = GithubRepo(repo)

    # Check if the target directory exists
    tool_path = tool_path or pathlib.Path(github_repo.repo)
    if tool_path.exists():
        if (
            Prompt.ask(f":axe: Overwrite {tool_path.absolute()}?", choices=["y", "n"], default="n")
            == "n"
        ):
            return
        rich.print()
        shutil.rmtree(tool_path)

    # Check if the repo is accessible
    if github_repo.exists:
        temp_dir = download_and_unzip_archive(github_repo.zip_url)

    # This could be a private repo that the user can access
    # by getting an access token from our API
    elif github_repo.namespace == "dreadnode":
        # Validate server configuration for private repository access
        user_config = UserConfig.read()
        profile_to_use = validate_server_for_clone(user_config, None)

        if profile_to_use is None:
            return  # User cancelled

        github_access_token = create_api_client(profile=profile_to_use).get_github_access_token(
            [github_repo.repo]
        )
        rich.print(":key: Accessed private repository")
        temp_dir = download_and_unzip_archive(
            github_repo.api_zip_url,
            headers={"Authorization": f"Bearer {github_access_token.token}"},
        )

    else:
        raise RuntimeError(f"Repository '{github_repo}' not found or inaccessible")

    # We assume the repo download results in a single
    # child folder which is the real target
    sub_dirs = list(temp_dir.iterdir())
    if len(sub_dirs) == 1 and sub_dirs[0].is_dir():
        temp_dir = sub_dirs[0]

    shutil.move(temp_dir, tool_path)

    rich.print()
    rich.print(f":tada: Cloned [b]{repo}[/] to [b]{tool_path.absolute()}[/]")


@cli.command()
async def install(tool: str) -> None:
    """Install a tool by name using its Ansible playbook."""

    from dreadnode.agent.tools.install.deps import install_tool

    try:
        install_tool(tool)
        rich.print(f":tada: Successfully installed tool '{tool}'.")
    except FileNotFoundError as e:
        rich.print(f":exclamation: {e}")
    except RuntimeError as e:
        rich.print(f":exclamation: {e}")


@cli.command()
async def run(
    tool: str,
    config: pathlib.Path | None = None,
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

    file_path: pathlib.Path | None = None
    tool_name: str | None = None

    if tool is not None:
        tool_name = tool
        tool_as_path = pathlib.Path(tool.split(":")[0]).with_suffix(".py")
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
        input: t.Annotated[str, cyclopts.Parameter(help="Input to the agent")],
        *,
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

    # command, bound, _ = tool_app.parse_args(tokens)

    # result = command(*bound.args, **bound.kwargs)
    # if isawaitable(result):
    #     await result
