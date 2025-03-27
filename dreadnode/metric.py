import inspect
import typing as t
from dataclasses import dataclass, field
from datetime import datetime, timezone

from logfire._internal.utils import safe_repr
from opentelemetry.trace import Tracer

from dreadnode.types import JsonDict, JsonValue

T = t.TypeVar("T")


@dataclass
class Metric:
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
        "Create a composite metric from individual values and weights."
        total = sum(value * weight for _, value, weight in values)
        weight = sum(weight for _, _, weight in values)
        score_attributes = {name: value for name, value, _ in values}
        return cls(value=total / weight, step=step, attributes={**attributes, **score_attributes})


MetricDict = dict[str, list[Metric]]

ScorerResult = float | int | bool | Metric
ScorerCallable = t.Callable[[T], t.Awaitable[ScorerResult]] | t.Callable[[T], ScorerResult]


@dataclass
class Scorer(t.Generic[T]):
    tracer: Tracer

    name: str
    tags: t.Sequence[str]
    attributes: dict[str, t.Any]
    func: ScorerCallable[T]

    @classmethod
    def from_callable(
        cls,
        tracer: Tracer,
        func: ScorerCallable[T] | "Scorer[T]",
        *,
        name: str | None = None,
        tags: t.Sequence[str] | None = None,
        attributes: dict[str, t.Any] | None = None,
    ) -> "Scorer[T]":
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
        self.__signature__ = inspect.signature(self.func)
        self.__name__ = self.name

    def clone(self) -> "Scorer[T]":
        return Scorer(
            tracer=self.tracer,
            name=self.name,
            tags=self.tags,
            attributes=self.attributes,
            func=self.func,
        )

    async def __call__(self, object: T) -> Metric:
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
                step=0,  # Do we integrate an increment/state system here?
                timestamp=datetime.now(timezone.utc),
                attributes=self.attributes,
            )

        return metric
