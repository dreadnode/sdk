"""AIRT (Attack) CLI for discovering and running attacks."""

import typing as t

from dreadnode.cli.discoverable import DiscoverableCLI

if t.TYPE_CHECKING:
    from dreadnode.airt.attack import Attack


async def _run_attack(attack: "Attack", input: str | None, raw: bool) -> None:
    """Run an attack."""
    await (attack.run() if raw else attack.console())


def _create_airt_cli() -> DiscoverableCLI["Attack"]:
    """Create the airt CLI using the shared discoverable pattern."""
    from dreadnode.airt.attack import Attack
    from dreadnode.core.optimization.format import format_studies, format_study

    return DiscoverableCLI(
        name="airt",
        discovery_type=Attack,
        help_text="Discover and run AIRT attacks.",
        object_name="attack",
        format_single=format_study,
        format_multiple=format_studies,
        get_object_name=lambda a: a.name,
        get_object_description=lambda a: a.description,
        run_object=_run_attack,
        requires_input=False,
    )


# Create the CLI app
_cli = _create_airt_cli()
airt_cli = _cli.app
