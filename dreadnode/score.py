import contextlib
import inspect
import typing as t
from dataclasses import dataclass
from datetime import datetime, timezone

from logfire._internal.utils import safe_repr
from opentelemetry.trace import Tracer

from .tracing import Metric, Span

T = t.TypeVar("T")

ScorerResult = float | int | bool | Metric
ScorerCallable = t.Callable[[T], ScorerResult | t.Awaitable[ScorerResult]]


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
        qualified_func_name = func_name = getattr(func, "__qualname__", getattr(func, "__name__", safe_repr(func)))
        with contextlib.suppress(Exception):
            qualified_func_name = f"{inspect.getmodule(func).__name__}.{func_name}"  # type: ignore
        name = name or str(getattr(func, "__name__", qualified_func_name))
        return cls(tracer=tracer, name=name, tags=tags or [], attributes=attributes or {}, func=func)

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
                step=0,  # TODO: Do we integrate an increment/state system here?
                timestamp=datetime.now(timezone.utc),
                attributes=self.attributes,
            )

        return metric
