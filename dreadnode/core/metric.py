import typing as t
from datetime import datetime, timezone

import typing_extensions as te
from pydantic import Field
from pydantic.dataclasses import dataclass

from dreadnode.core.exceptions import warn_at_user_stacklevel
from dreadnode.core.types.common import JsonDict, JsonValue

T = t.TypeVar("T")

MetricAggMode = t.Literal["avg", "sum", "min", "max", "count"]


MetricsDict = dict[str, "MetricSeries"]
MetricsLike = dict[str, float | bool] | list["MetricDict"]


class MetricWarning(UserWarning):
    """Warning for metrics-related issues"""


class MetricDict(te.TypedDict, total=False):
    """Dictionary representation of a metric for easier APIs"""

    name: str
    value: float | bool
    step: int
    timestamp: datetime | None
    aggregation: MetricAggMode | None
    attributes: JsonDict | None
    origin: t.Any | None


@dataclass
class MetricSeries:
    """
    A series of metric values with aggregation computed on read.

    This replaces dict[str, list[Metric]] for metric storage.
    Raw values are always preserved, and any aggregation can be
    computed at query time.

    Attributes:
        values: The raw metric values in order of logging.
        steps: Optional step indices for each value.
        timestamps: Timestamps for each value.
    """

    values: list[float] = Field(default_factory=list)
    steps: list[int | None] = Field(default_factory=list)
    timestamps: list[datetime] = Field(default_factory=list)

    def append(
        self,
        value: float,
        step: int | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Append a value to the series."""
        self.values.append(value)
        self.steps.append(step if step is not None else len(self.values) - 1)
        self.timestamps.append(timestamp or datetime.now(timezone.utc))

    # Aggregations - computed on read, raw data preserved

    def last(self) -> float | None:
        """Get the last value in the series."""
        return self.values[-1] if self.values else None

    def first(self) -> float | None:
        """Get the first value in the series."""
        return self.values[0] if self.values else None

    def mean(self) -> float | None:
        """Compute the mean of all values."""
        return sum(self.values) / len(self.values) if self.values else None

    def min(self) -> float | None:
        """Get the minimum value."""
        return min(self.values) if self.values else None

    def max(self) -> float | None:
        """Get the maximum value."""
        return max(self.values) if self.values else None

    def sum(self) -> float:
        """Get the sum of all values."""
        return sum(self.values)

    def count(self) -> int:
        """Get the number of values."""
        return len(self.values)

    def at_step(self, step: int) -> float | None:
        """Get the value at a specific step."""
        for i, s in enumerate(self.steps):
            if s == step:
                return self.values[i]
        return None

    def values_at_steps(self, steps: t.Sequence[int]) -> list[float | None]:
        """Get values at multiple steps."""
        return [self.at_step(s) for s in steps]

    @property
    def value(self) -> float | None:
        """Convenience property for single-value series (same as last)."""
        return self.last()

    def to_metric(self, aggregation: MetricAggMode = "avg") -> "Metric":
        """Convert to a single Metric using the specified aggregation."""
        if not self.values:
            raise ValueError("Cannot convert empty series to Metric")

        agg_value: float
        if aggregation == "avg":
            agg_value = self.mean() or 0.0
        elif aggregation == "sum":
            agg_value = self.sum()
        elif aggregation == "min":
            agg_value = self.min() or 0.0
        elif aggregation == "max":
            agg_value = self.max() or 0.0
        elif aggregation == "count":
            agg_value = float(self.count())
        else:
            agg_value = self.last() or 0.0

        return Metric(
            value=agg_value,
            step=self.steps[-1] or 0,
            timestamp=self.timestamps[-1],
            attributes={"aggregation": aggregation, "count": self.count()},
        )


@dataclass
class Metric:
    """
    Any reported value regarding the state of a run, task, and optionally object (input/output).

    Attributes:
        value: The value of the metric, e.g. 0.5, 1.0, 2.0, etc.
        step: An step value to indicate when this metric was reported.
        timestamp: The timestamp when the metric was reported.
        attributes: A dictionary of attributes to attach to the metric.
    """

    value: float
    step: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    attributes: JsonDict = Field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Metric(value={self.value}, step={self.step})"

    @classmethod
    def from_many(
        cls,
        values: t.Sequence[tuple[str, float, float]],
        step: int = 0,
        **attributes: JsonValue,
    ) -> "Metric":
        """
        Create a composite metric from individual values and weights.

        This is useful for creating a metric that is the weighted average of multiple values.
        The values should be a sequence of tuples, where each tuple contains the name of the metric,
        the value of the metric, and the weight of the metric.

        The individual values will be reported in the attributes of the metric.

        Args:
            values: A sequence of tuples containing the name, value, and weight of each metric.
            step: The step value to attach to the metric.
            **attributes: Additional attributes to attach to the metric.

        Returns:
            A composite Metric
        """
        total = sum(value * weight for _, value, weight in values)
        weight = sum(weight for _, _, weight in values)
        score_attributes = {name: value for name, value, _ in values}
        return cls(
            value=total / weight,
            step=step,
            attributes={**attributes, **score_attributes},
        )

    def apply_aggregation(self, agg: MetricAggMode, others: "list[Metric]") -> "Metric":
        """
        Apply an aggregation mode to the metric.
        This will modify the metric in place.

        Args:
            mode: The mode to apply. One of "sum", "min", "max", or "count".
            others: A list of other metrics to apply the mode to.

        Returns:
            self
        """
        previous_agg = next((m.attributes.get("aggregation") for m in others), agg)
        if previous_agg is not None and agg != previous_agg:
            warn_at_user_stacklevel(
                f"Metric logged with different aggregation ({agg} != {previous_agg}). This will result in unexpected behavior.",
                MetricWarning,
            )

        self.attributes["original"] = self.value
        self.attributes["agg"] = agg

        prior_values = [m.value for m in sorted(others, key=lambda m: m.timestamp)]

        if agg == "sum":
            self.value += max(prior_values) if prior_values else 0
        elif agg == "min":
            self.value = min([self.value, *prior_values])
        elif agg == "max":
            self.value = max([self.value, *prior_values])
        elif agg == "count":
            self.value = len(others) + 1
        elif agg == "avg" and prior_values:
            current_avg = prior_values[-1]
            self.value = current_avg + (self.value - current_avg) / (len(prior_values) + 1)

        return self
