import contextlib
import inspect
import itertools
import typing as t
from pathlib import Path

import cyclopts
import rich

from dreadnode.cli.shared import DreadnodeConfig
from dreadnode.discovery import DEFAULT_SEARCH_PATHS, discover
from dreadnode.meta import get_config_model, hydrate

cli = cyclopts.App("study", help="Discover and run evaluations.")


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
    Discover and list available studies in a Python file.

    If no file is specified, searches in standard paths.
    """
    from dreadnode.airt import Attack
    from dreadnode.optimization import Study
    from dreadnode.optimization.format import format_studies, format_study

    discovered = discover(Study, file, exclude_types={Attack})
    if not discovered:
        path_hint = file or ", ".join(DEFAULT_SEARCH_PATHS)
        rich.print(f"No studies found in {path_hint}.")
        return

    grouped_by_path = itertools.groupby(discovered, key=lambda a: a.path)
    for path, discovered_studies in grouped_by_path:
        studies = [study.obj for study in discovered_studies]
        rich.print(f"Studies in [bold]{path}[/bold]:\n")
        if verbose:
            for study in studies:
                rich.print(format_study(study))
        else:
            rich.print(format_studies(studies))


@cli.command()
async def run(  # noqa: PLR0912, PLR0915
    study: str,
    *tokens: t.Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    config: Path | None = None,
    raw: t.Annotated[bool, cyclopts.Parameter(["-r", "--raw"], negative=False)] = False,
    dn_config: DreadnodeConfig | None = None,
) -> None:
    """
    Run a study by name, file, or module.

    - If just a file is passed, it will search for the first study in that file ('my_studies.py').\n
    - If just a study name is passed, it will search for that study in the default files ('hyperparam_search').\n
    - If the study is specified with a file, it will run that specific study in the given file ('my_studies.py:hyperparam_search').\n
    - If the file is not specified, it defaults to searching for main.py, study.py, or app.py.

    **To get detailed help for a specific study, use `dreadnode study run <study> help`.**

    Args:
        study: The study to run, e.g., 'my_studies.py:hyperparam' or 'hyperparam'.
        config: Optional path to a TOML/YAML/JSON configuration file for the study.
        raw: If set, only display raw logging output without additional formatting.
    """
    from dreadnode.airt import Attack
    from dreadnode.optimization import Study

    file_path: Path | None = None
    study_name: str | None = None

    if study is not None:
        study_name = study
        study_as_path = Path(study.split(":")[0]).with_suffix(".py")
        if study_as_path.exists():
            file_path = study_as_path
            study_name = study.split(":", 1)[-1] if ":" in study else None

    path_hint = file_path or ", ".join(DEFAULT_SEARCH_PATHS)

    discovered = discover(Study, file_path, exclude_types={Attack})
    if not discovered:
        rich.print(f":exclamation: No studies found in {path_hint}.")
        return

    studies_by_name = {d.obj.name: d.obj for d in discovered}
    studies_by_lower_name = {k.lower(): v for k, v in studies_by_name.items()}
    if study_name is None:
        if len(discovered) > 1:
            rich.print(
                f"[yellow]Warning:[/yellow] Multiple studies found. Defaulting to the first one: '{next(iter(studies_by_name.keys()))}'."
            )
        study_name = next(iter(studies_by_name.keys()))

    if study_name.lower() not in studies_by_lower_name:
        rich.print(f":exclamation: Study '{study_name}' not found in {path_hint}.")
        rich.print(f"Available studies are: {', '.join(studies_by_name.keys())}")
        return

    study_blueprint = studies_by_lower_name[study_name.lower()]

    config_model = get_config_model(study_blueprint)
    config_annotation = cyclopts.Parameter(name="*", group="Study Config")(config_model)
    config_default: t.Any = inspect.Parameter.empty
    with contextlib.suppress(Exception):
        config_default = config_model()

    async def study_cli(
        *,
        config: t.Any = config_default,
        dn_config: DreadnodeConfig | None = dn_config,
    ) -> None:
        dn_config = dn_config or DreadnodeConfig()
        if raw and dn_config.log_level is None:
            dn_config.log_level = "info"
        dn_config.apply()

        study_obj = hydrate(study_blueprint, config)
        await (study_obj.run() if raw else study_obj.console())

    study_cli.__annotations__["config"] = config_annotation

    help_text = f"Run the '{study_name}' study."
    if study_blueprint.description:
        help_text += "\n\n" + study_blueprint.description

    study_app = cyclopts.App(
        name=study_name,
        help=help_text,
        help_on_error=True,
        help_flags=("help"),
        version_flags=(),
    )
    study_app.default(study_cli)

    if config:
        if not config.exists():
            rich.print(f":exclamation: Configuration file '{config}' does not exist.")
            return

        if config.suffix in {".toml"}:
            study_app._config = cyclopts.config.Toml(config, use_commands_as_keys=False)  # type: ignore[assignment] # noqa: SLF001
        elif config.suffix in {".yaml", ".yml"}:
            study_app._config = cyclopts.config.Yaml(config, use_commands_as_keys=False)  # type: ignore[assignment] # noqa: SLF001
        elif config.suffix in {".json"}:
            study_app._config = cyclopts.config.Json(config, use_commands_as_keys=False)  # type: ignore[assignment] # noqa: SLF001
        else:
            rich.print(f":exclamation: Unsupported configuration file format: '{config.suffix}'.")
            return

    command, bound, _ = study_app.parse_args(tokens)

    result = command(*bound.args, **bound.kwargs)
    if inspect.isawaitable(result):
        await result
