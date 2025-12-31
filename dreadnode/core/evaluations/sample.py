import typing as t
from datetime import datetime

import typing_extensions as te
from pydantic import BaseModel, ConfigDict, Field, computed_field
from ulid import ULID

from dreadnode.core.exceptions import AssertionFailedError
from dreadnode.core.metric import MetricSeries
from dreadnode.core.scorer import ScorerResult
from dreadnode.core.tracing.span import TaskSpan
from dreadnode.core.types.common import UNSET, ErrorField

if t.TYPE_CHECKING:
    from dreadnode.core.task import Task
    from dreadnode.core.types.common import AnyDict

In = te.TypeVar("In", default=t.Any)
Out = te.TypeVar("Out", default=t.Any)

FileFormat = t.Literal["jsonl", "csv", "json", "yaml", "yml"]

InputDataset = list[In]
InputDatasetProcessor = t.Callable[[InputDataset], InputDataset]


class Sample(BaseModel, t.Generic[In, Out]):
    """
    Represents a single input-output sample processed by a task,

    along with associated metadata such as metrics, assertions, and context.

    Attributes:
        id: Unique identifier for the sample.
        input: The sample input value.
        output: The sample output value.
        index: The index of the sample in the dataset.
        iteration: The iteration this sample belongs to.
        scenario_params: The parameters defining the scenario this sample belongs to.
        metrics: Metrics collected during measurement.
        assertions: Assertions made during measurement.
        context: Contextual information about the sample - like originating dataset fields.
        error: Any error that occurred.
        task: Associated task span.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    id: ULID = Field(default_factory=ULID)
    input: In
    output: Out | None = None
    index: int = 0
    iteration: int = 0
    scenario_params: dict[str, t.Any] = Field(default_factory=dict)
    metrics: dict[str, MetricSeries] = Field(default_factory=dict)  # observations
    scores: dict[str, ScorerResult] = Field(default_factory=dict)  # evaluations (separate!)
    assertions: dict[str, bool] = Field(default_factory=dict)
    context: dict[str, t.Any] | None = Field(default=None, repr=False)
    error: ErrorField | None = Field(default=None, repr=False)
    task: TaskSpan[Out] | None = Field(default=None, repr=False)

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

    def get_average_metric_value(self, key: str) -> float:
        """
        Computes the average value of the specified metric.

        Args:
            key: The key of the metric to average.
        """
        series = self.metrics.get(key)
        return series.mean() if series else 0.0

    @classmethod
    def from_task(
        cls,
        task: "Task[..., t.Any]",
        span: TaskSpan[Out],
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

        for name, series in self.metrics.items():
            if series.value is not None:
                record[f"metric_{name}"] = series.mean()

        return record
