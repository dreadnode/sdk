"""Evaluation CLI for discovering and running evaluations."""

import typing as t

from dreadnode.cli.discoverable import DiscoverableCLI

if t.TYPE_CHECKING:
    from dreadnode.evaluations import Evaluation


async def _run_evaluation(evaluation: "Evaluation", input: str | None, raw: bool) -> None:
    """Run an evaluation."""
    await (evaluation.run() if raw else evaluation.console())


def _create_evaluation_cli() -> DiscoverableCLI["Evaluation"]:
    """Create the evaluation CLI using the shared discoverable pattern."""
    from dreadnode.core.evaluations.format import format_eval, format_evals
    from dreadnode.evaluations import Evaluation

    return DiscoverableCLI(
        name="eval",
        discovery_type=Evaluation,
        help_text="Discover and run evaluations.",
        object_name="evaluation",
        format_single=format_eval,
        format_multiple=format_evals,
        get_object_name=lambda e: e.name,
        get_object_description=lambda e: e.description,
        run_object=_run_evaluation,
        requires_input=False,
    )


# Create the CLI app
_cli = _create_evaluation_cli()
evaluation_cli = _cli.app
