import contextlib
import inspect
import typing as t
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import groupby

from logfire._internal.utils import safe_repr
from opentelemetry.trace import Tracer

from .tracing import Metric, MetricDict, Score, Span

T = t.TypeVar("T")

ScorerCallable = t.Callable[[T], float | Score | t.Awaitable[float | Score]]


@dataclass
class Scorer(t.Generic[T]):
    tracer: Tracer

    name: str
    attributes: dict[str, t.Any]
    func: ScorerCallable[T]

    @classmethod
    def from_callable(
        cls,
        tracer: Tracer,
        func: ScorerCallable[T] | "Scorer[T]",
        *,
        name: str | None = None,
        attributes: dict[str, t.Any] | None = None,
    ) -> "Scorer[T]":
        if isinstance(func, Scorer):
            if name is not None or attributes is not None:
                func = func.clone()
                func.name = name or func.name
                func.attributes.update(attributes or {})
            return func

        func = inspect.unwrap(func)
        qualified_func_name = func_name = getattr(func, "__qualname__", getattr(func, "__name__", safe_repr(func)))
        with contextlib.suppress(Exception):
            qualified_func_name = f"{inspect.getmodule(func).__name__}.{func_name}"  # type: ignore
        name = name or str(getattr(func, "__name__", qualified_func_name))
        return cls(tracer=tracer, name=name, attributes=attributes or {}, func=func)

    def __post_init__(self) -> None:
        self.__signature__ = inspect.signature(self.func)
        self.__name__ = self.name

    def clone(self) -> "Scorer[T]":
        return Scorer(
            tracer=self.tracer,
            name=self.name,
            attributes=self.attributes,
            func=self.func,
        )

    async def __call__(self, object: T) -> Score:
        with Span(
            name=self.name,
            attributes=self.attributes,
            tracer=self.tracer,
        ):
            score = self.func(object)
            if inspect.isawaitable(score):
                score = await score

        if not isinstance(score, Score):
            score = Score(timestamp=datetime.now(), name=self.name, value=float(score))

        return score


def scores_to_metrics(scores: t.Sequence[Score]) -> MetricDict:
    if not scores:
        return {}

    sorted_by_name = sorted(scores, key=lambda score: score.name)
    grouped_by_name = {name: list(group) for name, group in groupby(sorted_by_name, key=lambda score: score.name)}

    metrics: MetricDict = defaultdict(list)
    for name, group in grouped_by_name.items():
        score_group = sorted(group, key=lambda score: score.timestamp)

        # Write the flat scores inside a single metric + cumulative
        cumulative: float = 0
        for score in score_group:
            metric = Metric(timestamp=score.timestamp, value=score.value, step=0)
            metrics[name].append(metric)
            cumulative += score.value
            metrics[f"{name}.cum"].append(Metric(timestamp=score.timestamp, value=cumulative, step=0))

        if not score_group:
            continue

        # Include an average
        avg_value = sum(score.value for score in score_group) / len(score_group)
        avg_metric = Metric(timestamp=datetime.now(), value=avg_value, step=0)
        metrics[f"{name}.avg"].append(avg_metric)

        # Include a count
        count_metric = Metric(timestamp=datetime.now(), value=float(len(score_group)), step=0)
        metrics[f"{name}.count"].append(count_metric)

    return dict(metrics)
