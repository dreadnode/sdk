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

cli = cyclopts.App("attack", help="Discover and run AIRT attacks.")


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
    Discover and list available attacks in a Python file.

    If no file is specified, searches in common paths.
    """
    from dreadnode.airt import Attack
    from dreadnode.optimization.format import format_studies, format_study

    discovered = discover(Attack, file)
    if not discovered:
        path_hint = file or ", ".join(DEFAULT_SEARCH_PATHS)
        rich.print(f"No attacks found in {path_hint}.")
        return

    grouped_by_path = itertools.groupby(discovered, key=lambda a: a.path)
    for path, discovered_attacks in grouped_by_path:
        attacks = [attack.obj for attack in discovered_attacks]
        rich.print(f"Attacks in [bold]{path}[/bold]:\n")
        if verbose:
            for attack in attacks:
                rich.print(format_study(attack))
        else:
            rich.print(format_studies(attacks))


@cli.command()
async def run(  # noqa: PLR0912, PLR0915
    attack: str,
    *tokens: t.Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
    config: Path | None = None,
    raw: t.Annotated[bool, cyclopts.Parameter(["-r", "--raw"], negative=False)] = False,
    dn_config: DreadnodeConfig | None = None,
) -> None:
    """
    Run a attack by name, file, or module.

    - If just a file is passed, it will search for the first attack in that file ('my_attacks.py').\n
    - If just a attack name is passed, it will search for that attack in the default files ('hopskip').\n
    - If the attack is specified with a file, it will run that specific attack in the given file ('my_attacks.py:hopskip').\n
    - If the file is not specified, it defaults to searching in common paths.

    **To get detailed help for a specific attack, use `dreadnode attack run <attack> help`.**

    Args:
        attack: The attack to run, e.g., 'my_attacks.py:hopskip' or 'hopskip'.
        config: Optional path to a TOML/YAML/JSON configuration file for the attack.
        raw: If set, only display raw logging output without additional formatting.
    """
    from dreadnode.airt import Attack

    file_path: Path | None = None
    attack_name: str | None = None

    if attack is not None:
        attack_name = attack
        attack_as_path = Path(attack.split(":")[0]).with_suffix(".py")
        if attack_as_path.exists():
            file_path = attack_as_path
            attack_name = attack.split(":", 1)[-1] if ":" in attack else None

    path_hint = file_path or ", ".join(DEFAULT_SEARCH_PATHS)

    discovered = discover(Attack, file_path)
    if not discovered:
        rich.print(f":exclamation: No attacks found in {path_hint}.")
        return

    attacks_by_name = {d.obj.name: d.obj for d in discovered}
    attacks_by_lower_name = {k.lower(): v for k, v in attacks_by_name.items()}
    if attack_name is None:
        if len(discovered) > 1:
            rich.print(
                f"[yellow]Warning:[/yellow] Multiple attacks found. Defaulting to the first one: '{next(iter(attacks_by_name.keys()))}'."
            )
        attack_name = next(iter(attacks_by_name.keys()))

    if attack_name.lower() not in attacks_by_lower_name:
        rich.print(f":exclamation: Attack '{attack_name}' not found in {path_hint}.")
        rich.print(f"Available attacks are: {', '.join(attacks_by_name.keys())}")
        return

    attack_blueprint = attacks_by_lower_name[attack_name.lower()]

    config_model = get_config_model(attack_blueprint)
    config_annotation = cyclopts.Parameter(name="*", group="Attack Config")(config_model)
    config_default: t.Any = inspect.Parameter.empty
    with contextlib.suppress(Exception):
        config_default = config_model()

    async def attack_cli(
        *,
        config: t.Any = config_default,
        dn_config: DreadnodeConfig | None = dn_config,
    ) -> None:
        dn_config = dn_config or DreadnodeConfig()
        if raw and dn_config.log_level is None:
            dn_config.log_level = "info"
        dn_config.apply()

        attack_obj = hydrate(attack_blueprint, config)
        await (attack_obj.run() if raw else attack_obj.console())

    attack_cli.__annotations__["config"] = config_annotation

    help_text = f"Run the '{attack_name}' attack."
    if attack_blueprint.description:
        help_text += "\n\n" + attack_blueprint.description

    attack_app = cyclopts.App(
        name=attack_name,
        help=help_text,
        help_on_error=True,
        help_flags=("help"),
        version_flags=(),
    )
    attack_app.default(attack_cli)

    if config:
        if not config.exists():
            rich.print(f":exclamation: Configuration file '{config}' does not exist.")
            return

        if config.suffix in {".toml"}:
            attack_app._config = cyclopts.config.Toml(config, use_commands_as_keys=False)  # type: ignore[assignment] # noqa: SLF001
        elif config.suffix in {".yaml", ".yml"}:
            attack_app._config = cyclopts.config.Yaml(config, use_commands_as_keys=False)  # type: ignore[assignment] # noqa: SLF001
        elif config.suffix in {".json"}:
            attack_app._config = cyclopts.config.Json(config, use_commands_as_keys=False)  # type: ignore[assignment] # noqa: SLF001
        else:
            rich.print(f":exclamation: Unsupported configuration file format: '{config.suffix}'.")
            return

    command, bound, _ = attack_app.parse_args(tokens)

    result = command(*bound.args, **bound.kwargs)
    if inspect.isawaitable(result):
        await result
