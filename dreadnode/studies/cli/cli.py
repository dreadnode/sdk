"""Study CLI for discovering and running optimization studies."""

import typing as t

from dreadnode.cli.discoverable import DiscoverableCLI

if t.TYPE_CHECKING:
    from dreadnode.core.optimization import Study


async def _run_study(study: "Study", input: str | None, raw: bool) -> None:
    """Run a study."""
    await (study.run() if raw else study.console())


def _create_study_cli() -> DiscoverableCLI["Study"]:
    """Create the study CLI using the shared discoverable pattern."""
    from dreadnode.airt.attack import Attack
    from dreadnode.core.optimization import Study
    from dreadnode.core.optimization.format import format_studies, format_study

    # Create a custom CLI that excludes Attack subclass
    cli = DiscoverableCLI(
        name="study",
        discovery_type=Study,
        help_text="Discover and run optimization studies.",
        object_name="study",
        format_single=format_study,
        format_multiple=format_studies,
        get_object_name=lambda s: s.name,
        get_object_description=lambda s: s.description,
        run_object=_run_study,
        requires_input=False,
    )

    # Override discovery to exclude Attack subclass
    def discover_and_find_excluding_attacks(identifier: str) -> tuple["Study | None", str, str]:
        from dreadnode.core.discovery import DEFAULT_SEARCH_PATHS, discover

        import rich

        file_path, obj_name = cli._parse_identifier(identifier)
        path_hint = str(file_path) if file_path else ", ".join(DEFAULT_SEARCH_PATHS)

        # Discover studies but exclude Attack
        discovered = discover(Study, file_path, exclude_types={Attack})
        if not discovered:
            return None, obj_name or "", path_hint

        objs_by_name = {cli.get_object_name(d.obj): d.obj for d in discovered}
        objs_by_lower_name = {k.lower(): v for k, v in objs_by_name.items()}

        if obj_name is None:
            if len(discovered) > 1:
                rich.print(
                    f"[yellow]Warning:[/yellow] Multiple studies found. "
                    f"Defaulting to the first one: '{next(iter(objs_by_name.keys()))}'."
                )
            obj_name = next(iter(objs_by_name.keys()))

        if obj_name.lower() not in objs_by_lower_name:
            rich.print(f":exclamation: Study '{obj_name}' not found in {path_hint}.")
            rich.print(f"Available studies are: {', '.join(objs_by_name.keys())}")
            return None, obj_name, path_hint

        return objs_by_lower_name[obj_name.lower()], obj_name, path_hint

    cli._discover_and_find = discover_and_find_excluding_attacks

    return cli


# Create the CLI app
_cli = _create_study_cli()
study_cli = _cli.app
