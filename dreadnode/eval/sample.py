import typing as t
from datetime import datetime

import typing_extensions as te
from pydantic import BaseModel, ConfigDict, Field, computed_field
from ulid import ULID

from dreadnode.common_types import UNSET, ErrorField
from dreadnode.error import AssertionFailedError
from dreadnode.metric import Metric
from dreadnode.tracing.span import TaskSpan

if t.TYPE_CHECKING:
    from dreadnode.common_types import AnyDict
    from dreadnode.task import Task

In = te.TypeVar("In", default=t.Any)
Out = te.TypeVar("Out", default=t.Any)

FileFormat = t.Literal["jsonl", "csv", "json", "yaml", "yml"]

InputDataset = list[In]
InputDatasetProcessor = t.Callable[[InputDataset], InputDataset]


class Sample(BaseModel, t.Generic[In, Out]):
    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    id: ULID = Field(default_factory=ULID)
    """Unique identifier for the sample."""

    input: In
    """The sample input value."""
    output: Out | None = None
    """The sample output value."""
    transformed_input: In | None = None
    """The input after transforms were applied (if any)."""
    index: int = 0
    """The index of the sample in the dataset."""
    iteration: int = 0
    """The iteration this sample belongs to."""
    scenario_params: dict[str, t.Any] = Field(default_factory=dict)
    """The parameters defining the scenario this sample belongs to."""

    metrics: dict[str, list[Metric]] = Field(default_factory=dict)
    """Metrics collected during measurement."""
    assertions: dict[str, bool] = Field(default_factory=dict)
    """Assertions made during measurement."""
    context: dict[str, t.Any] | None = Field(default=None, repr=False)
    """Contextual information about the sample - like originating dataset fields."""
    error: ErrorField | None = Field(default=None, repr=False)
    """Any error that occurred."""
    task: TaskSpan[Out] | None = Field(default=None, repr=False)
    """Associated task span."""

    @computed_field  # type: ignore[prop-decorator]
    @property
    def created_at(self) -> datetime:
        """The creation timestamp of the sample, extracted from its ULID."""
        return self.id.datetime

    @property
    def passed(self) -> bool:
        """Whether all assertions have passed."""
        return all(self.assertions.values()) if self.assertions else True

    @property
    def failed(self) -> bool:
        """Whether the underlying task failed for reasons other than score assertions."""
        return self.error is not None and not isinstance(self.error, AssertionFailedError)

    def get_average_metric_value(self, key: str | None = None) -> float:
        """
        Computes the average value of the specified metric across all samples.

        Args:
            key: The key of the metric to average. If None, averages all metrics.
        """
        metrics = (
            self.metrics.get(key, [])
            if key is not None
            else [m for ms in self.metrics.values() for m in ms]
        )

        if not metrics:
            return 0.0

        return sum(metric.value for metric in metrics) / len(metrics)

    @classmethod
    def from_task(
        cls,
        task: "Task[..., t.Any]",  # The configured task that was run.
        span: TaskSpan[Out],  # The resulting span from the run.
        input: t.Any,
        *,
        scenario_params: dict[str, t.Any] | None = None,
        iteration: int = 0,
        index: int = 0,
        context: dict[str, t.Any] | None = None,
    ) -> "Sample[In, Out]":
        # Assume false for all
        assert_scores: t.Any = getattr(task, "assert_scores", [])
        assertions = dict.fromkeys(assert_scores, False)

        # If a score was reported, assume true
        for name in set(span.metrics.keys()) & set(assertions.keys()):
            assertions[name] = True

        # Reset to false for any that triggered a failure
        if isinstance(span.exception, AssertionFailedError):
            for name in span.exception.failures:
                assertions[name] = False

        output: Out | None = None
        if span._output is not UNSET:  # noqa: SLF001
            output = t.cast("Out", span._output)  # noqa: SLF001

        return cls(
            input=t.cast("In", input),
            output=output,
            index=index,
            iteration=iteration,
            scenario_params=scenario_params or {},
            metrics=span.metrics,
            assertions=assertions,
            context=context,
            error=span.exception,
            task=span,  # The sample is associated with the span, not the task blueprint.
        )

    def to_dict(self) -> dict[str, t.Any]:
        """
        Flattens the sample's data, performing necessary transformations
        (like metric pivoting) suitable for DataFrame conversion.
        """
        record: AnyDict = self.model_dump(
            exclude={"metrics", "assertions", "task"},
            mode="json",
        )

        record["passed"] = self.passed
        record["failed"] = self.failed
        record["task"] = self.task.name if self.task else None

        for name, value in record.pop("scenario_params", {}).items():
            record[f"param_{name}"] = value

        for assertion_name, passed in self.assertions.items():
            record[f"assertion_{assertion_name}"] = passed

        record_inputs = record.get("input", {})
        if isinstance(record_inputs, dict):
            for name, value in record_inputs.items():
                record[f"input_{name}"] = value

        for name, metrics in self.metrics.items():
            if metrics:
                avg_value = sum(m.value for m in metrics) / len(metrics)
                record[f"metric_{name}"] = avg_value

        return record
