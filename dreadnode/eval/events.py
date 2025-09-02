import typing as t
from dataclasses import dataclass, field

import typing_extensions as te

if t.TYPE_CHECKING:
    from dreadnode.eval.eval import Eval
    from dreadnode.eval.result import EvalResult, IterationResult, ScenarioResult
    from dreadnode.eval.sample import Sample

In = te.TypeVar("In", default=t.Any)
Out = te.TypeVar("Out", default=t.Any)

EvalStopReason = t.Literal["finished", "max_consecutive_failures_reached"]


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
    stop_reason: EvalStopReason = "finished"


def rebuild_event_models() -> None:
    pass
    # from dreadnode.eval.eval import Eval
    # from dreadnode.eval.result import EvalResult, IterationResult, ScenarioResult
    # from dreadnode.eval.sample import Sample

    # rebuild_dataclass(EvalEvent)  # type: ignore[arg-type]
    # rebuild_dataclass(EvalStart)  # type: ignore[arg-type]
    # rebuild_dataclass(EvalEnd)  # type: ignore[arg-type]
    # rebuild_dataclass(ScenarioStart)  # type: ignore[arg-type]
    # rebuild_dataclass(ScenarioEnd)  # type: ignore[arg-type]
    # rebuild_dataclass(IterationStart)  # type: ignore[arg-type]
    # rebuild_dataclass(IterationEnd)  # type: ignore[arg-type]
    # rebuild_dataclass(SampleComplete)  # type: ignore[arg-type]
