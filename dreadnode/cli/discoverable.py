"""
Shared utilities for creating discovery-based CLI applications.

This module provides factory functions and utilities to reduce code duplication
across the various discoverable object CLIs (agents, evaluations, studies, attacks).
"""

import contextlib
import inspect
import itertools
import typing as t
from inspect import isawaitable
from pathlib import Path

import cyclopts
import rich

from dreadnode.cli.shared import DreadnodeConfig
from dreadnode.core.discovery import DEFAULT_SEARCH_PATHS, discover
from dreadnode.core.log import console
from dreadnode.core.meta import get_config_model, hydrate
from dreadnode.core.meta.introspect import flatten_model

T = t.TypeVar("T")

# Type aliases for formatter functions
SingleFormatter = t.Callable[[T], t.Any]
MultiFormatter = t.Callable[[list[T]], t.Any]


class DiscoverableCLI(t.Generic[T]):
    """
    A factory class for creating discovery-based CLI applications.

    This eliminates code duplication across agent, evaluation, study, and attack CLIs
    by providing a common pattern for:
    - Listing/showing discovered objects
    - Running discovered objects with dynamic configuration
    """

    def __init__(
        self,
        name: str,
        discovery_type: type[T],
        *,
        help_text: str,
        object_name: str,  # e.g., "agent", "evaluation", "study", "attack"
        format_single: SingleFormatter[T],
        format_multiple: MultiFormatter[T],
        get_object_name: t.Callable[[T], str],
        get_object_description: t.Callable[[T], str | None],
        run_object: t.Callable[[T, str | None, bool], t.Awaitable[None]],
        requires_input: bool = False,
    ):
        """
        Create a discoverable CLI.

        Args:
            name: The CLI command name (e.g., "agent", "eval", "study")
            discovery_type: The type to discover (e.g., Agent, Evaluation, Study)
            help_text: Help text for the CLI app
            object_name: Human-readable name for the object type
            format_single: Function to format a single object for verbose display
            format_multiple: Function to format multiple objects for list display
            get_object_name: Function to get the name of an object
            get_object_description: Function to get the description of an object
            run_object: Async function to run an object (obj, input, raw) -> None
            requires_input: Whether the run command requires an input argument
        """
        self.name = name
        self.discovery_type = discovery_type
        self.help_text = help_text
        self.object_name = object_name
        self.format_single = format_single
        self.format_multiple = format_multiple
        self.get_object_name = get_object_name
        self.get_object_description = get_object_description
        self.run_object = run_object
        self.requires_input = requires_input

        self.app = cyclopts.App(name, help=help_text)
        self._register_commands()

    def _register_commands(self) -> None:
        """Register the show and run commands."""

        @self.app.command(name=["list", "ls", "show"])
        def show(
            file: Path | None = None,
            *,
            verbose: t.Annotated[
                bool,
                cyclopts.Parameter(["--verbose", "-v"], help="Display detailed information."),
            ] = False,
        ) -> None:
            f"""
            Discover and list available {self.object_name}s in a Python file.

            If no file is specified, searches in standard paths.
            """
            discovered = discover(self.discovery_type, file)
            if not discovered:
                path_hint = file or ", ".join(DEFAULT_SEARCH_PATHS)
                rich.print(f"No {self.object_name}s found in {path_hint}.")
                return

            grouped_by_path = itertools.groupby(discovered, key=lambda a: a.path)
            for path, discovered_objs in grouped_by_path:
                objs = [d.obj for d in discovered_objs]
                rich.print(f"{self.object_name.capitalize()}s in [bold]{path}[/bold]:\n")
                if verbose:
                    for obj in objs:
                        rich.print(self.format_single(obj))
                else:
                    rich.print(self.format_multiple(objs))

        # Store reference to self for use in nested function
        cli_self = self

        if self.requires_input:
            @self.app.command()
            async def run(
                identifier: str,
                *tokens: t.Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
                config: Path | None = None,
                raw: t.Annotated[bool, cyclopts.Parameter(["-r", "--raw"], negative=False)] = False,
                dn_config: DreadnodeConfig | None = None,
            ) -> None:
                f"""
                Run a {cli_self.object_name} by name, file, or module.

                - If just a file is passed, it will search for the first {cli_self.object_name} in that file.\n
                - If just a name is passed, it will search for that {cli_self.object_name} in the default files.\n
                - If specified with a file, it will run that specific {cli_self.object_name} (e.g., 'file.py:name').\n
                - If the file is not specified, it defaults to searching standard paths.

                **To get detailed help, use `dreadnode {cli_self.name} run <{cli_self.object_name}> help`.**

                Args:
                    identifier: The {cli_self.object_name} to run, e.g., 'my_file.py:name' or 'name'.
                    config: Optional path to a TOML/YAML/JSON configuration file.
                    raw: If set, only display raw logging output without additional formatting.
                """
                await cli_self._run_with_input(identifier, tokens, config, raw, dn_config)
        else:
            @self.app.command()
            async def run(
                identifier: str,
                *tokens: t.Annotated[str, cyclopts.Parameter(show=False, allow_leading_hyphen=True)],
                config: Path | None = None,
                raw: t.Annotated[bool, cyclopts.Parameter(["-r", "--raw"], negative=False)] = False,
                dn_config: DreadnodeConfig | None = None,
            ) -> None:
                f"""
                Run a {cli_self.object_name} by name, file, or module.

                - If just a file is passed, it will search for the first {cli_self.object_name} in that file.\n
                - If just a name is passed, it will search for that {cli_self.object_name} in the default files.\n
                - If specified with a file, it will run that specific {cli_self.object_name} (e.g., 'file.py:name').\n
                - If the file is not specified, it defaults to searching standard paths.

                **To get detailed help, use `dreadnode {cli_self.name} run <{cli_self.object_name}> help`.**

                Args:
                    identifier: The {cli_self.object_name} to run, e.g., 'my_file.py:name' or 'name'.
                    config: Optional path to a TOML/YAML/JSON configuration file.
                    raw: If set, only display raw logging output without additional formatting.
                """
                await cli_self._run_without_input(identifier, tokens, config, raw, dn_config)

    def _parse_identifier(self, identifier: str) -> tuple[Path | None, str | None]:
        """Parse an identifier into file path and object name."""
        file_path: Path | None = None
        obj_name: str | None = identifier

        if identifier is not None:
            path_part = identifier.split(":")[0]
            candidate_path = Path(path_part).with_suffix(".py")
            if candidate_path.exists():
                file_path = candidate_path
                obj_name = identifier.split(":", 1)[-1] if ":" in identifier else None

        return file_path, obj_name

    def _discover_and_find(
        self, identifier: str
    ) -> tuple[T | None, str, str]:
        """
        Discover objects and find the requested one.

        Returns:
            Tuple of (found_object, object_name, path_hint)
        """
        file_path, obj_name = self._parse_identifier(identifier)
        path_hint = str(file_path) if file_path else ", ".join(DEFAULT_SEARCH_PATHS)

        discovered = discover(self.discovery_type, file_path)
        if not discovered:
            return None, obj_name or "", path_hint

        objs_by_name = {self.get_object_name(d.obj): d.obj for d in discovered}
        objs_by_lower_name = {k.lower(): v for k, v in objs_by_name.items()}

        if obj_name is None:
            if len(discovered) > 1:
                rich.print(
                    f"[yellow]Warning:[/yellow] Multiple {self.object_name}s found. "
                    f"Defaulting to the first one: '{next(iter(objs_by_name.keys()))}'."
                )
            obj_name = next(iter(objs_by_name.keys()))

        if obj_name.lower() not in objs_by_lower_name:
            rich.print(f":exclamation: {self.object_name.capitalize()} '{obj_name}' not found in {path_hint}.")
            rich.print(f"Available {self.object_name}s are: {', '.join(objs_by_name.keys())}")
            return None, obj_name, path_hint

        return objs_by_lower_name[obj_name.lower()], obj_name, path_hint

    def _setup_config_file(self, app: cyclopts.App, config: Path) -> bool:
        """Set up configuration file for the app. Returns False if config is invalid."""
        if not config.exists():
            rich.print(f":exclamation: Configuration file '{config}' does not exist.")
            return False

        if config.suffix == ".toml":
            app._config = cyclopts.config.Toml(config, use_commands_as_keys=False)  # type: ignore[assignment] # noqa: SLF001
        elif config.suffix in {".yaml", ".yml"}:
            app._config = cyclopts.config.Yaml(config, use_commands_as_keys=False)  # type: ignore[assignment] # noqa: SLF001
        elif config.suffix == ".json":
            app._config = cyclopts.config.Json(config, use_commands_as_keys=False)  # type: ignore[assignment] # noqa: SLF001
        else:
            rich.print(f":exclamation: Unsupported configuration file format: '{config.suffix}'.")
            return False

        return True

    async def _run_with_input(
        self,
        identifier: str,
        tokens: tuple[str, ...],
        config_file: Path | None,
        raw: bool,
        dn_config: DreadnodeConfig | None,
    ) -> None:
        """Run an object that requires an input argument (like agents)."""
        blueprint, obj_name, path_hint = self._discover_and_find(identifier)
        if blueprint is None:
            if not obj_name:
                rich.print(f":exclamation: No {self.object_name}s found in {path_hint}.")
            return

        config_model = get_config_model(blueprint)
        config_annotation = cyclopts.Parameter(name="*", group=f"{self.object_name.capitalize()} Config")(config_model)
        config_default: t.Any = inspect.Parameter.empty
        with contextlib.suppress(Exception):
            config_default = config_model()

        run_object = self.run_object
        get_name = self.get_object_name
        object_name = self.object_name

        async def inner_cli(
            input: t.Annotated[str, cyclopts.Parameter(help=f"Input to the {object_name}")],
            *,
            config: t.Any = config_default,
            dn_config: DreadnodeConfig | None = dn_config,
        ) -> None:
            dn_config = dn_config or DreadnodeConfig()
            if raw and dn_config.log_level is None:
                dn_config.log_level = "info"
            dn_config.apply()

            obj = hydrate(blueprint, config)

            rich.print(f"Running {object_name}: [bold]{get_name(obj)}[/bold] with config:")
            for key, value in flatten_model(config).items():
                rich.print(f" |- {key}: {value}")
            rich.print()

            await run_object(obj, input, raw)

        inner_cli.__annotations__["config"] = config_annotation

        help_text = f"Run the '{obj_name}' {self.object_name}."
        description = self.get_object_description(blueprint)
        if description:
            help_text += "\n\n" + description

        inner_app = cyclopts.App(
            name=obj_name,
            help=help_text,
            help_on_error=True,
            help_flags=("help",),
            version_flags=(),
            console=console,
        )
        inner_app.default(inner_cli)

        if config_file and not self._setup_config_file(inner_app, config_file):
            return

        command, bound, _ = inner_app.parse_args(tokens)
        result = command(*bound.args, **bound.kwargs)
        if isawaitable(result):
            await result

    async def _run_without_input(
        self,
        identifier: str,
        tokens: tuple[str, ...],
        config_file: Path | None,
        raw: bool,
        dn_config: DreadnodeConfig | None,
    ) -> None:
        """Run an object that doesn't require an input argument (like evaluations, studies)."""
        blueprint, obj_name, path_hint = self._discover_and_find(identifier)
        if blueprint is None:
            if not obj_name:
                rich.print(f":exclamation: No {self.object_name}s found in {path_hint}.")
            return

        config_model = get_config_model(blueprint)
        config_annotation = cyclopts.Parameter(name="*", group=f"{self.object_name.capitalize()} Config")(config_model)
        config_default: t.Any = inspect.Parameter.empty
        with contextlib.suppress(Exception):
            config_default = config_model()

        run_object = self.run_object

        async def inner_cli(
            *,
            config: t.Any = config_default,
            dn_config: DreadnodeConfig | None = dn_config,
        ) -> None:
            dn_config = dn_config or DreadnodeConfig()
            if raw and dn_config.log_level is None:
                dn_config.log_level = "info"
            dn_config.apply()

            obj = hydrate(blueprint, config)
            await run_object(obj, None, raw)

        inner_cli.__annotations__["config"] = config_annotation

        help_text = f"Run the '{obj_name}' {self.object_name}."
        description = self.get_object_description(blueprint)
        if description:
            help_text += "\n\n" + description

        inner_app = cyclopts.App(
            name=obj_name,
            help=help_text,
            help_on_error=True,
            help_flags=("help",),
            version_flags=(),
            console=console,
        )
        inner_app.default(inner_cli)

        if config_file and not self._setup_config_file(inner_app, config_file):
            return

        command, bound, _ = inner_app.parse_args(tokens)
        result = command(*bound.args, **bound.kwargs)
        if isawaitable(result):
            await result
