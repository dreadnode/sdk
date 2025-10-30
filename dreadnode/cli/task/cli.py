import contextlib
import inspect
import itertools
import typing as t
from inspect import isawaitable
from pathlib import Path
from textwrap import dedent

import cyclopts
import rich

from dreadnode.cli.shared import DreadnodeConfig
from dreadnode.discovery import DEFAULT_SEARCH_PATHS, discover
from dreadnode.meta import get_config_model, hydrate
from dreadnode.meta.introspect import flatten_model
from dreadnode.task import Task

cli = cyclopts.App("task", help="Discover and run tasks.")


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
    Discover and list available entrypoint tasks in a Python file.

    If no file is specified, searches in standard paths.
    """
    from dreadnode.format import format_task, format_tasks

    discovered = [d for d in discover(Task, file) if d.obj.entrypoint]
    if not discovered:
        path_hint = file or ", ".join(DEFAULT_SEARCH_PATHS)
        rich.print(f"No entrypoint tasks found in {path_hint}.")
        return

    grouped_by_path = itertools.groupby(discovered, key=lambda a: a.path)
    for path, discovered_tasks in grouped_by_path:
        tasks = [task.obj for task in discovered_tasks]
        rich.print(f"Tasks in [bold]{path}[/bold]:\n")
        if verbose:
            for task in tasks:
                rich.print(format_task(task))
        else:
            rich.print(format_tasks(tasks))


@cli.command()
async def run(  # noqa: PLR0912, PLR0915
    task: str,
    *tokens: t.Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    config: Path | None = None,
    dn_config: DreadnodeConfig | None = None,
) -> None:
    """
    Run a task by name, file, or module.

    - If just a file is passed, it will search for the first entrypoint task in that file ('my_tasks.py').
    - If just a task name is passed, it will search for that task in the default files ('process_data').
    - If the task is specified with a file, it will run that specific task in the given file ('my_tasks.py:process_data').
    - If the file is not specified, it defaults to searching for main.py, agent.py, app.py, or task.py.

    **To get detailed help for a specific task, use `dreadnode task run <task> help`.**

    Args:
        task: The task to run, e.g., 'my_tasks.py:process' or 'process'.
        config: Optional path to a TOML/YAML/JSON configuration file for the task.
    """
    file_path: Path | None = None
    task_name: str | None = None

    if task is not None:
        task_name = task
        task_as_path = Path(task.split(":")[0]).with_suffix(".py")
        if task_as_path.exists():
            file_path = task_as_path
            task_name = task.split(":", 1)[-1] if ":" in task else None

    path_hint = file_path or ", ".join(DEFAULT_SEARCH_PATHS)

    discovered = [d for d in discover(Task, file_path) if d.obj.entrypoint]
    if not discovered:
        rich.print(f":exclamation: No entrypoint tasks found in {path_hint}.")
        return

    tasks_by_name = {d.obj.name: d.obj for d in discovered}
    tasks_by_lower_name = {k.lower(): v for k, v in tasks_by_name.items()}
    if task_name is None:
        if len(discovered) > 1:
            rich.print(
                f"[yellow]Warning:[/yellow] Multiple tasks found. Defaulting to the first one: '{next(iter(tasks_by_name.keys()))}'."
            )
        task_name = next(iter(tasks_by_name.keys()))

    if task_name.lower() not in tasks_by_lower_name:
        rich.print(f":exclamation: Task '{task_name}' not found in '{path_hint}'.")
        rich.print(f"Available tasks are: {', '.join(tasks_by_name.keys())}")
        return

    task_blueprint = tasks_by_lower_name[task_name.lower()]

    config_model = get_config_model(task_blueprint)
    config_annotation = cyclopts.Parameter(name="*", group="Task Config")(config_model)
    config_default: t.Any = inspect.Parameter.empty
    with contextlib.suppress(Exception):
        config_default = config_model()

    async def task_cli(
        *,
        config: t.Any = config_default,
        dn_config: DreadnodeConfig | None = dn_config,
    ) -> None:
        (dn_config or DreadnodeConfig()).apply()

        hydrated_task = hydrate(task_blueprint, config)
        flat_config = flatten_model(config)

        rich.print(f"Running task: [bold]{hydrated_task.name}[/bold] with config:")
        for key, value in flat_config.items():
            rich.print(f" |- {key}: {value}")
        rich.print()

        await hydrated_task()

    task_cli.__annotations__["config"] = config_annotation

    help_text = f"Run the '{task_name}' task."
    if task_blueprint.__doc__:
        help_text += "\n\n" + dedent(task_blueprint.__doc__)

    task_app = cyclopts.App(
        name=task_name,
        help=help_text,
        help_on_error=True,
        help_flags=("help"),
        version_flags=(),
    )
    task_app.default(task_cli)

    if config:
        if not config.exists():
            rich.print(f":exclamation: Configuration file '{config}' does not exist.")
            return

        if config.suffix in {".toml"}:
            task_app._config = cyclopts.config.Toml(config, use_commands_as_keys=False)  # type: ignore[assignment] # noqa: SLF001
        elif config.suffix in {".yaml", ".yml"}:
            task_app._config = cyclopts.config.Yaml(config, use_commands_as_keys=False)  # type: ignore[assignment] # noqa: SLF001
        elif config.suffix in {".json"}:
            task_app._config = cyclopts.config.Json(config, use_commands_as_keys=False)  # type: ignore[assignment] # noqa: SLF001
        else:
            rich.print(f":exclamation: Unsupported configuration file format: '{config.suffix}'.")
            return

    command, bound, _ = task_app.parse_args(tokens)

    result = command(*bound.args, **bound.kwargs)
    if isawaitable(result):
        await result
