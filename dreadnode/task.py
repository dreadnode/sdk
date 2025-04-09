import asyncio
import inspect
import traceback
import typing as t
from dataclasses import dataclass

from logfire._internal.stack_info import warn_at_user_stacklevel
from opentelemetry.trace import Tracer

from dreadnode.metric import Scorer, ScorerCallable
from dreadnode.tracing.span import TaskSpan, current_run_span

P = t.ParamSpec("P")
R = t.TypeVar("R")


class TaskFailedWarning(UserWarning):
    pass


class TaskGeneratorWarning(UserWarning):
    pass


class TaskSpanList(list[TaskSpan[R]]):
    def sorted(self, *, reverse: bool = True) -> "TaskSpanList[R]":
        return TaskSpanList(
            sorted(self, key=lambda span: span.get_average_metric_value(), reverse=reverse),
        )

    @t.overload
    def top_n(
        self,
        n: int,
        *,
        as_outputs: t.Literal[False] = False,
        reverse: bool = True,
    ) -> "TaskSpanList[R]":
        ...

    @t.overload
    def top_n(self, n: int, *, as_outputs: t.Literal[True], reverse: bool = True) -> list[R]:
        ...

    def top_n(
        self,
        n: int,
        *,
        as_outputs: bool = False,
        reverse: bool = True,
    ) -> "TaskSpanList[R] | list[R]":
        sorted_ = self.sorted(reverse=reverse)[:n]
        return (
            t.cast(list[R], [span.output for span in sorted_])
            if as_outputs
            else TaskSpanList(sorted_)
        )


@dataclass
class Task(t.Generic[P, R]):
    tracer: Tracer

    name: str
    label: str
    attributes: dict[str, t.Any]
    func: t.Callable[P, R]
    scorers: list[Scorer[R]]
    tags: list[str]

    log_params: t.Sequence[str] | t.Literal[True] | None = None
    log_inputs: t.Sequence[str] | t.Literal[True] | None = None
    log_output: bool = True

    def __post_init__(self) -> None:
        self.__signature__ = inspect.signature(self.func)
        self.__name__ = getattr(self.func, "__name__", self.name)

    def _bind_args(self, *args: P.args, **kwargs: P.kwargs) -> dict[str, t.Any]:
        signature = inspect.signature(self.func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return dict(bound_args.arguments)

    def clone(self) -> "Task[P, R]":
        return Task(
            tracer=self.tracer,
            name=self.name,
            label=self.label,
            attributes=self.attributes.copy(),
            func=self.func,
            scorers=self.scorers.copy(),
            tags=self.tags.copy(),
        )

    def with_(
        self,
        *,
        scorers: t.Sequence[Scorer[R] | ScorerCallable[R]] | None = None,
        name: str | None = None,
        tags: t.Sequence[str] | None = None,
        label: str | None = None,
        append: bool = False,
        **attributes: t.Any,
    ) -> "Task[P, R]":
        task = self.clone()
        task.name = name or task.name
        task.label = label or task.label

        new_scorers = [Scorer.from_callable(self.tracer, scorer) for scorer in (scorers or [])]
        new_tags = list(tags or [])

        if append:
            task.scorers.extend(new_scorers)
            task.tags.extend(new_tags)
            task.attributes.update(attributes)
        else:
            task.scorers = new_scorers
            task.tags = new_tags
            task.attributes = attributes

        return task

    async def run(self, *args: P.args, **kwargs: P.kwargs) -> TaskSpan[R]:
        run = current_run_span.get()
        if run is None or not run.is_recording:
            raise RuntimeError("Tasks must be executed within a run")

        bound_args = self._bind_args(*args, **kwargs)

        params = (
            bound_args
            if self.log_params is True
            else {k: v for k, v in bound_args.items() if k in (self.log_params or [])}
        )
        inputs = (
            bound_args
            if self.log_inputs is True
            else {k: v for k, v in bound_args.items() if k in (self.log_inputs or [])}
        )

        with TaskSpan[R](
            name=self.name,
            label=self.label,
            attributes=self.attributes,
            params=params,
            tags=self.tags,
            run_id=run.run_id,
            tracer=self.tracer,
        ) as span:
            for name, value in inputs.items():
                span.log_input(name, value, label=f"{self.label}.input.{name}")

            output = t.cast(R | t.Awaitable[R], self.func(*args, **kwargs))
            if inspect.isawaitable(output):
                output = await output

            span.output = output

            if self.log_output:
                span.log_output("output", output, label=f"{self.label}.output")

            for scorer in self.scorers:
                metric = await scorer(span.output)
                span.log_metric(scorer.name, metric, origin=span.output)

        return span

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        span = await self.run(*args, **kwargs)
        return span.output

    # NOTE(nick): Not sure I'm in love with these being instance methods here.
    # We could move them to the top level class maybe.

    async def map_run(self, count: int, *args: P.args, **kwargs: P.kwargs) -> TaskSpanList[R]:
        spans = await asyncio.gather(*[self.run(*args, **kwargs) for _ in range(count)])
        return TaskSpanList(spans)

    async def map(self, count: int, *args: P.args, **kwargs: P.kwargs) -> list[R]:
        spans = await self.map_run(count, *args, **kwargs)
        return [span.output for span in spans]

    async def top_n(self, count: int, n: int, *args: P.args, **kwargs: P.kwargs) -> list[R]:
        spans = await self.map_run(count, *args, **kwargs)
        return spans.top_n(n, as_outputs=True)

    async def try_run(self, *args: P.args, **kwargs: P.kwargs) -> TaskSpan[R] | None:
        try:
            return await self.run(*args, **kwargs)
        except Exception:  # noqa: BLE001
            warn_at_user_stacklevel(
                f"Task '{self.name}' ({self.label}) failed:\n{traceback.format_exc()}",
                TaskFailedWarning,
            )
            return None

    async def try_(self, *args: P.args, **kwargs: P.kwargs) -> R | None:
        span = await self.try_run(*args, **kwargs)
        return span.output if span else None

    async def try_map_run(self, count: int, *args: P.args, **kwargs: P.kwargs) -> TaskSpanList[R]:
        spans = await asyncio.gather(*[self.try_run(*args, **kwargs) for _ in range(count)])
        return TaskSpanList([span for span in spans if span])

    async def try_top_n(self, count: int, n: int, *args: P.args, **kwargs: P.kwargs) -> list[R]:
        spans = await self.try_map_run(count, *args, **kwargs)
        return spans.top_n(n, as_outputs=True)

    async def try_map(self, count: int, *args: P.args, **kwargs: P.kwargs) -> list[R]:
        spans = await self.try_map_run(count, *args, **kwargs)
        return [span.output for span in spans if span]
