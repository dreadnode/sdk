"""
This module provides classes and methods for managing and reporting metrics.

Metrics are used to track the state of a run, task, or object (input/output).
The `Metric` class represents a single metric, while the `Scorer` class is used
to generate metrics from callable functions.

Classes:
    Metric: Represents a single metric with value, step, timestamp, and attributes.
    Scorer: Generates metrics from callable functions.

Exceptions:
    MetricWarning: A warning class for metric-related issues.

Types:
    MetricAggMode: Literal type for aggregation modes.
    MetricDict: Dictionary type for storing metrics.
    ScorerResult: Union type for scorer results.
    ScorerCallable: Callable type for scorer functions.
"""

import inspect
import typing as t
from dataclasses import dataclass, field
from datetime import datetime, timezone

from logfire._internal.stack_info import warn_at_user_stacklevel
from logfire._internal.utils import safe_repr
from opentelemetry.trace import Tracer

from dreadnode.types import JsonDict, JsonValue

T = t.TypeVar("T")

MetricAggMode = t.Literal["avg", "sum", "min", "max", "count"]


class MetricWarning(UserWarning):
    """A warning class for metric-related issues."""


@dataclass
class Metric:
    """
    Represents a single metric with value, step, timestamp, and attributes.

    Attributes:
        value (float): The value of the metric, e.g., 0.5, 1.0, 2.0, etc.
        step (int): The step value to indicate when this metric was reported.
        timestamp (datetime): The timestamp when the metric was reported.
        attributes (JsonDict): A dictionary of attributes to attach to the metric.
    """

    value: float
    step: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attributes: JsonDict = field(default_factory=dict)

    @classmethod
    def from_many(
        cls,
        values: t.Sequence[tuple[str, float, float]],
        step: int = 0,
        **attributes: JsonValue,
    ) -> "Metric":
        """
        Create a composite metric from individual values and weights.

        Args:
            values (Sequence[tuple[str, float, float]]): A sequence of tuples containing
                the name, value, and weight of each metric.
            step (int): The step value to attach to the metric.
            **attributes: Additional attributes to attach to the metric.

        Returns:
            Metric: A composite metric.
        """
        total = sum(value * weight for _, value, weight in values)
        weight = sum(weight for _, _, weight in values)
        score_attributes = {name: value for name, value, _ in values}
        return cls(value=total / weight, step=step, attributes={**attributes, **score_attributes})

    def apply_mode(self, mode: MetricAggMode, others: "list[Metric]") -> "Metric":
        """
        Apply an aggregation mode to the metric.

        This modifies the metric in place based on the specified mode.

        Args:
            mode (MetricAggMode): The mode to apply. One of "sum", "min", "max", "count", or "avg".
            others (list[Metric]): A list of other metrics to apply the mode to.

        Returns:
            Metric: The modified metric.
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
            self.value += max(prior_values)
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


MetricDict = dict[str, list[Metric]]

ScorerResult = float | int | bool | Metric
ScorerCallable = t.Callable[[T], t.Awaitable[ScorerResult]] | t.Callable[[T], ScorerResult]


@dataclass
class Scorer(t.Generic[T]):
    """
    Generates metrics from callable functions.

    Attributes:
        tracer (Tracer): The tracer to use for reporting metrics.
        name (str): The name of the scorer, used for reporting metrics.
        tags (Sequence[str]): A list of tags to attach to the metric.
        attributes (dict[str, Any]): A dictionary of attributes to attach to the metric.
        func (ScorerCallable[T]): The function to call to get the metric.
        step (int): The step value to attach to metrics produced by this Scorer.
        auto_increment_step (bool): Whether to automatically increment the step for each call.
    """

    tracer: Tracer
    name: str
    tags: t.Sequence[str]
    attributes: dict[str, t.Any]
    func: ScorerCallable[T]
    step: int = 0
    auto_increment_step: bool = False

    @classmethod
    def from_callable(
        cls,
        tracer: Tracer,
        func: "ScorerCallable[T] | Scorer[T]",
        *,
        name: str | None = None,
        tags: t.Sequence[str] | None = None,
        **attributes: t.Any,
    ) -> "Scorer[T]":
        """
        Create a scorer from a callable function.

        Args:
            tracer (Tracer): The tracer to use for reporting metrics.
            func (ScorerCallable[T] | Scorer[T]): The function to call to get the metric.
            name (str | None): The name of the scorer, used for reporting metrics.
            tags (Sequence[str] | None): A list of tags to attach to the metric.
            **attributes: Additional attributes to attach to the metric.

        Returns:
            Scorer[T]: A Scorer object.
        """
        if isinstance(func, Scorer):
            if name is not None or attributes is not None:
                func = func.clone()
                func.name = name or func.name
                func.attributes.update(attributes or {})
            return func

        func = inspect.unwrap(func)
        func_name = getattr(
            func,
            "__qualname__",
            getattr(func, "__name__", safe_repr(func)),
        )
        name = name or func_name
        return cls(
            tracer=tracer,
            name=name,
            tags=tags or [],
            attributes=attributes or {},
            func=func,
        )

    def __post_init__(self) -> None:
        """Initialize the scorer's signature and name."""
        self.__signature__ = inspect.signature(self.func)
        self.__name__ = self.name

    def clone(self) -> "Scorer[T]":
        """
        Clone the scorer.

        Returns:
            Scorer[T]: A new Scorer object with the same attributes.
        """
        return Scorer(
            tracer=self.tracer,
            name=self.name,
            tags=self.tags,
            attributes=self.attributes,
            func=self.func,
            step=self.step,
            auto_increment_step=self.auto_increment_step,
        )

    async def __call__(self, object: T) -> Metric:
        """
        Execute the scorer and return the metric.

        Args:
            object (T): The object to score.

        Returns:
            Metric: A Metric object representing the result of the scoring function.
        """
        from dreadnode.tracing.span import Span

        with Span(
            name=self.name,
            tags=self.tags,
            attributes=self.attributes,
            tracer=self.tracer,
        ):
            metric = self.func(object)
            if inspect.isawaitable(metric):
                metric = await metric

        if not isinstance(metric, Metric):
            metric = Metric(
                float(metric),
                step=self.step,
                timestamp=datetime.now(timezone.utc),
                attributes=self.attributes,
            )

        if self.auto_increment_step:
            self.step += 1

        return metric
