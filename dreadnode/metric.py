import typing as t
from datetime import datetime, timezone

import typing_extensions as te
from pydantic import Field
from pydantic.dataclasses import dataclass

from dreadnode.common_types import JsonDict, JsonValue
from dreadnode.util import warn_at_user_stacklevel

T = t.TypeVar("T")

MetricAggMode = t.Literal["avg", "sum", "min", "max", "count"]
"""
Aggregation modes for metrics:"
- "avg": Average of the values.
- "sum": Sum of the values.
- "min": Minimum value.
- "max": Maximum value.
- "count": Count of the values.
"""

MetricsDict = dict[str, "list[Metric]"]
"""A dictionary of metrics, where the key is the metric name and the value is a list of metrics with that name."""
MetricsLike = dict[str, float | bool] | list["MetricDict"]
"""
Either a dictionary of metric names to values (float or bool) or a list of metric dictionaries.

Examples:
- `{"accuracy": 0.95, "loss": 0.05}`
- `[{"name": "accuracy", "value": 0.95}, {"name": "loss", "value": 0.05}]`
"""


class MetricWarning(UserWarning):
    """Warning for metrics-related issues"""


class MetricDict(te.TypedDict, total=False):
    """Dictionary representation of a metric for easier APIs"""

    name: str
    value: float | bool
    step: int
    timestamp: datetime | None
    mode: MetricAggMode | None
    attributes: JsonDict | None
    origin: t.Any | None


@dataclass
class Metric:
    """
    Any reported value regarding the state of a run, task, and optionally object (input/output).
    """

    value: float
    "The value of the metric, e.g. 0.5, 1.0, 2.0, etc."
    step: int = 0
    "An step value to indicate when this metric was reported."
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    "The timestamp when the metric was reported."
    attributes: JsonDict = Field(default_factory=dict)
    "A dictionary of attributes to attach to the metric."

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

    def apply_mode(self, mode: MetricAggMode, others: "list[Metric]") -> "Metric":
        """
        Apply an aggregation mode to the metric.
        This will modify the metric in place.

        Args:
            mode: The mode to apply. One of "sum", "min", "max", or "count".
            others: A list of other metrics to apply the mode to.

        Returns:
            self
        """
        previous_mode = next((m.attributes.get("mode") for m in others), mode)
        if previous_mode is not None and mode != previous_mode:
            warn_at_user_stacklevel(
                f"Metric logged with different modes ({mode} != {previous_mode}). This may result in unexpected behavior.",
                MetricWarning,
            )

        self.attributes["original"] = self.value
        self.attributes["mode"] = mode

        prior_values = [m.value for m in sorted(others, key=lambda m: m.timestamp)]

        if mode == "sum":
            # Take the max of the priors because they might already be summed
            self.value += max(prior_values) if prior_values else 0
        elif mode == "min":
            self.value = min([self.value, *prior_values])
        elif mode == "max":
            self.value = max([self.value, *prior_values])
        elif mode == "count":
            self.value = len(others) + 1
        elif mode == "avg" and prior_values:
            current_avg = prior_values[-1]
            self.value = current_avg + (self.value - current_avg) / (len(prior_values) + 1)

        return self
