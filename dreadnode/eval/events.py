import typing as t
from dataclasses import dataclass, field

import typing_extensions as te

if t.TYPE_CHECKING:
    from dreadnode.eval.eval import Eval
    from dreadnode.eval.result import EvalResult, IterationResult, ScenarioResult
    from dreadnode.eval.sample import Sample

In = te.TypeVar("In", default=t.Any)
Out = te.TypeVar("Out", default=t.Any)


@dataclass
class EvalEvent(t.Generic[In, Out]):
    """Base class for all evaluation events."""

    eval: "Eval[In, Out]" = field(repr=False)


@dataclass
class EvalStart(EvalEvent[In, Out]):
    """Signals the beginning of an evaluation."""

    dataset_size: int
    scenario_count: int
    total_iterations: int
    total_samples: int


@dataclass
class EvalEventInRun(EvalEvent[In, Out]):
    """Base class for all evaluation events that occur within a specific run."""

    run_id: str


@dataclass
class ScenarioStart(EvalEventInRun[In, Out]):
    """Signals the start of a new scenario."""

    scenario_params: dict[str, t.Any]
    iteration_count: int
    sample_count: int


@dataclass
class IterationStart(EvalEventInRun[In, Out]):
    """Signals the start of a new iteration within a scenario."""

    scenario_params: dict[str, t.Any]
    iteration: int


@dataclass
class SampleComplete(EvalEventInRun[In, Out]):
    """Signals that a single sample has completed processing."""

    sample: "Sample[In, Out]"


@dataclass
class IterationEnd(EvalEventInRun[In, Out]):
    """Signals the end of an iteration, containing its aggregated result."""

    result: "IterationResult[In, Out]"


@dataclass
class ScenarioEnd(EvalEventInRun[In, Out]):
    """Signals the end of a scenario, containing its aggregated result."""

    result: "ScenarioResult[In, Out]"


@dataclass
class EvalEnd(EvalEvent[In, Out]):
    """Signals the end of the entire evaluation, containing the final result."""

    result: "EvalResult[In, Out]"


@dataclass
class SamplePreProcess(EvalEventInRun[In, Out]):
    """Event before sample processing (hook point for input transforms)."""

    index: int
    dataset_row: dict[str, t.Any]
    task_kwargs: dict[str, t.Any]
    original_input: In


@dataclass
class SamplePostProcess(EvalEventInRun[In, Out]):
    """Event after sample processing (hook point for output transforms)."""

    index: int
    output: Out | None
    error: Exception | None
