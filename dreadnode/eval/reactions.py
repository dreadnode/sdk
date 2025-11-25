import typing as t
from dataclasses import dataclass


@dataclass
class EvalReaction(Exception):  # noqa: N818
    """Base class for evaluation reactions."""


@dataclass
class ModifyInput(EvalReaction):
    """Modify task input arguments."""

    task_kwargs: dict[str, t.Any]


@dataclass
class ModifyOutput(EvalReaction):
    """Modify task output."""

    output: t.Any


@dataclass
class SkipSample(EvalReaction):
    """Skip processing this sample."""

    reason: str


@dataclass
class StopEval(EvalReaction):
    """Stop the entire evaluation."""

    reason: str
