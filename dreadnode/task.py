import asyncio
import inspect
import typing as t
from dataclasses import dataclass

from opentelemetry.trace import Tracer

from .score import Scorer, ScorerCallable
from .tracing import TaskSpan, current_run_span

P = t.ParamSpec("P")
R = t.TypeVar("R")


class TaskSpanList(list[TaskSpan[R]]):
    def sorted(self, *, reverse: bool = True) -> "TaskSpanList[R]":
        return TaskSpanList(sorted(self, key=lambda span: span.average_score, reverse=reverse))

    @t.overload
    def top_n(self, n: int, *, as_outputs: t.Literal[False] = False, reverse: bool = True) -> "TaskSpanList[R]":
        ...

    @t.overload
    def top_n(self, n: int, *, as_outputs: t.Literal[True], reverse: bool = True) -> list[R]:
        ...

    def top_n(self, n: int, *, as_outputs: bool = False, reverse: bool = True) -> "TaskSpanList[R] | list[R]":
        sorted = self.sorted(reverse=reverse)[:n]
        return t.cast(list[R], [span.output for span in sorted]) if as_outputs else TaskSpanList(sorted)


@dataclass
class Task(t.Generic[P, R]):
    tracer: Tracer

    name: str
    attributes: dict[str, t.Any]
    func: t.Callable[P, t.Awaitable[R]]
    scorers: list[Scorer[R]]
    tags: list[str]

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
        append: bool = False,
        **attributes: t.Any,
    ) -> "Task[P, R]":
        task = self.clone()
        task.name = name or task.name

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

        with TaskSpan[R](
            name=self.name,
            attributes=self.attributes,
            args=self._bind_args(*args, **kwargs),
            run_id=run.run_id,
            tracer=self.tracer,
        ) as span:
            output = t.cast(R | t.Awaitable[R], self.func(*args, **kwargs))
            if inspect.isawaitable(output):
                output = await output

            span.output = output

            for scorer in self.scorers:
                score = await scorer(span.output)
                span.scores.append(score)
                run.scores.append(score)

        return span

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        span = await self.run(*args, **kwargs)
        return span.output  # type: ignore

    # TODO: Not sure I'm in love with these being instance methods here.
    # We could move them to the top level class maybe.

    async def map_run(self, count: int, *args: P.args, **kwargs: P.kwargs) -> TaskSpanList[R]:
        spans = await asyncio.gather(*[self.run(*args, **kwargs) for _ in range(count)])
        return TaskSpanList(spans)

    async def map(self, count: int, *args: P.args, **kwargs: P.kwargs) -> list[R]:
        spans = await self.map_run(count, *args, **kwargs)
        return [span.output for span in spans]  # type: ignore

    async def top_n(self, count: int, n: int, *args: P.args, **kwargs: P.kwargs) -> list[R]:
        spans = await self.map_run(count, *args, **kwargs)
        return spans.top_n(n, as_outputs=True)

    async def try_run(self, *args: P.args, **kwargs: P.kwargs) -> TaskSpan[R] | None:
        try:
            return await self.run(*args, **kwargs)
        except Exception as e:  # TODO: Pretty this up
            print(f"Task {self.name} failed with exception: {e}")
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
        return [span.output for span in spans if span]  # type: ignore
