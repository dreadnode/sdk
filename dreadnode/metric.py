import inspect
import typing as t
from dataclasses import dataclass, field
from datetime import datetime, timezone

import typing_extensions as te
from logfire._internal.stack_info import warn_at_user_stacklevel
from logfire._internal.utils import safe_repr

from dreadnode.types import JsonDict, JsonValue

T = t.TypeVar("T")

MetricAggMode = t.Literal["avg", "sum", "min", "max", "count"]


class MetricWarning(UserWarning):
    pass


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
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    "The timestamp when the metric was reported."
    attributes: JsonDict = field(default_factory=dict)
    "A dictionary of attributes to attach to the metric."

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


MetricsDict = dict[str, list[Metric]]
"""A dictionary of metrics, where the key is the metric name and the value is a list of metrics with that name."""
ScorerResult = float | int | bool | Metric
"""The result of a scorer function, which can be a numeric value or a Metric object."""
ScorerCallable = t.Callable[[T], t.Awaitable[ScorerResult]] | t.Callable[[T], ScorerResult]


@dataclass
class Scorer(t.Generic[T]):
    name: str
    "The name of the scorer, used for reporting metrics."
    tags: t.Sequence[str]
    "A list of tags to attach to the metric."
    attributes: dict[str, t.Any]
    "A dictionary of attributes to attach to the metric."
    func: ScorerCallable[T]
    "The function to call to get the metric."
    step: int = 0
    "The step value to attach to metrics produced by this Scorer."
    auto_increment_step: bool = False
    "Whether to automatically increment the step for each time this scorer is called."
    catch: bool = False
    "Whether to catch exceptions in the scorer function and return a 0 Metric with error information."

    @classmethod
    def from_callable(
        cls,
        func: "ScorerCallable[T] | Scorer[T]",
        *,
        name: str | None = None,
        tags: t.Sequence[str] | None = None,
        catch: bool = False,
        **attributes: t.Any,
    ) -> "Scorer[T]":
        """
        Create a scorer from a callable function.

        Args:
            func: The function to call to get the metric.
            name: The name of the scorer, used for reporting metrics.
            tags: A list of tags to attach to the metric.
            catch: Whether to catch exceptions in the scorer function and return a 0 Metric with error information.
            **attributes: A dictionary of attributes to attach to the metric.

        Returns:
            A Scorer object.
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
            name=name,
            tags=tags or [],
            attributes=attributes or {},
            func=func,
            catch=catch,
        )

    def __post_init__(self) -> None:
        self.__signature__ = inspect.signature(self.func)
        self.__name__ = self.name

    def clone(self) -> "Scorer[T]":
        """
        Clone the scorer.

        Returns:
            A new Scorer.
        """
        return Scorer(
            name=self.name,
            tags=self.tags,
            attributes=self.attributes,
            func=self.func,
            step=self.step,
            auto_increment_step=self.auto_increment_step,
            catch=self.catch,
        )

    async def __call__(self, object: T) -> Metric:
        """
        Execute the scorer and return the metric.

        Any output value will be converted to a Metric object.

        Args:
            object: The object to score.

        Returns:
            A Metric object.
        """
        try:
            metric = self.func(object)
            if inspect.isawaitable(metric):
                metric = await metric
        except Exception as exc:
            if not self.catch:
                raise

            warn_at_user_stacklevel(
                f"Error executing scorer {self.name!r} for object {object!r}: {exc}",
                MetricWarning,
            )
            metric = Metric(value=0.0, step=self.step, attributes={"error": str(exc)})

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
