import contextlib
import inspect
import itertools
import typing as t
from pathlib import Path

import cyclopts
import rich

from dreadnode.cli.shared import DreadnodeConfig
from dreadnode.discovery import DEFAULT_SEARCH_PATHS, discover
from dreadnode.logging_ import console as logging_console
from dreadnode.meta import get_config_model, hydrate

cli = cyclopts.App("eval", help="Discover and run evaluations.")


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
    Discover and list available evals in a Python file.

    If no file is specified, searches in standard paths.
    """
    from dreadnode.eval import Eval
    from dreadnode.eval.format import format_eval, format_evals

    discovered = discover(Eval, file)
    if not discovered:
        path_hint = file or ", ".join(DEFAULT_SEARCH_PATHS)
        rich.print(f"No evals found in {path_hint}.")
        return

    grouped_by_path = itertools.groupby(discovered, key=lambda a: a.path)
    for path, discovered_evals in grouped_by_path:
        evals = [eval_obj.obj for eval_obj in discovered_evals]
        rich.print(f"Evals in [bold]{path}[/bold]:\n")
        if verbose:
            for eval_obj in evals:
                rich.print(format_eval(eval_obj))
        else:
            rich.print(format_evals(evals))


@cli.command()
async def run(  # noqa: PLR0912, PLR0915
    evaluation: str,
    *tokens: t.Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    config: Path | None = None,
    raw: t.Annotated[bool, cyclopts.Parameter(["-r", "--raw"], negative=False)] = False,
    dn_config: DreadnodeConfig | None = None,
) -> None:
    """
    Run an eval by name, file, or module.

    - If just a file is passed, it will search for the first eval in that file ('my_evals.py').\n
    - If just an eval name is passed, it will search for that eval in the default files ('accuracy_test').\n
    - If the eval is specified with a file, it will run that specific eval in the given file ('my_evals.py:accuracy_test').\n
    - If the file is not specified, it defaults to searching for main.py, eval.py, or app.py.

    **To get detailed help for a specific eval, use `dreadnode eval run <eval> help`.**

    Args:
        eval: The eval to run, e.g., 'my_evals.py:accuracy' or 'accuracy'.
        config: Optional path to a TOML/YAML/JSON configuration file for the eval.
        raw: If set, only display raw logging output without additional formatting.
    """
    from dreadnode.eval import Eval

    file_path: Path | None = None
    eval_name: str | None = None

    if evaluation is not None:
        eval_name = evaluation
        eval_as_path = Path(evaluation.split(":")[0]).with_suffix(".py")
        if eval_as_path.exists():
            file_path = eval_as_path
            eval_name = evaluation.split(":", 1)[-1] if ":" in evaluation else None

    path_hint = file_path or ", ".join(DEFAULT_SEARCH_PATHS)

    discovered = discover(Eval, file_path)
    if not discovered:
        rich.print(f":exclamation: No evals found in {path_hint}.")
        return

    evals_by_name = {d.obj.name: d.obj for d in discovered}
    evals_by_lower_name = {k.lower(): v for k, v in evals_by_name.items()}
    if eval_name is None:
        if len(discovered) > 1:
            rich.print(
                f"[yellow]Warning:[/yellow] Multiple evals found. Defaulting to the first one: '{next(iter(evals_by_name.keys()))}'."
            )
        eval_name = next(iter(evals_by_name.keys()))

    if eval_name.lower() not in evals_by_lower_name:
        rich.print(f":exclamation: Eval '{eval_name}' not found in {path_hint}.")
        rich.print(f"Available evals are: {', '.join(evals_by_name.keys())}")
        return

    eval_blueprint = evals_by_lower_name[eval_name.lower()]

    config_model = get_config_model(eval_blueprint)
    config_annotation = cyclopts.Parameter(name="*", group="Eval Config")(config_model)
    config_default: t.Any = inspect.Parameter.empty
    with contextlib.suppress(Exception):
        config_default = config_model()

    async def eval_cli(
        *,
        config: t.Any = config_default,
        dn_config: DreadnodeConfig | None = dn_config,
    ) -> None:
        dn_config = dn_config or DreadnodeConfig()
        if raw and dn_config.log_level is None:
            dn_config.log_level = "info"
        dn_config.apply()

        eval_obj = hydrate(eval_blueprint, config)
        await (eval_obj.run() if raw else eval_obj.console())

    eval_cli.__annotations__["config"] = config_annotation

    help_text = f"Run the '{eval_name}' eval."
    if eval_blueprint.description:
        help_text += "\n\n" + eval_blueprint.description

    eval_app = cyclopts.App(
        name=eval_name,
        help=help_text,
        help_on_error=True,
        help_flags=("help"),
        version_flags=(),
        console=logging_console,
    )
    eval_app.default(eval_cli)

    if config:
        if not config.exists():
            rich.print(f":exclamation: Configuration file '{config}' does not exist.")
            return

        if config.suffix in {".toml"}:
            eval_app._config = cyclopts.config.Toml(config, use_commands_as_keys=False)  # type: ignore[assignment] # noqa: SLF001
        elif config.suffix in {".yaml", ".yml"}:
            eval_app._config = cyclopts.config.Yaml(config, use_commands_as_keys=False)  # type: ignore[assignment] # noqa: SLF001
        elif config.suffix in {".json"}:
            eval_app._config = cyclopts.config.Json(config, use_commands_as_keys=False)  # type: ignore[assignment] # noqa: SLF001
        else:
            rich.print(f":exclamation: Unsupported configuration file format: '{config.suffix}'.")
            return

    command, bound, _ = eval_app.parse_args(tokens)

    result = command(*bound.args, **bound.kwargs)
    if inspect.isawaitable(result):
        await result
