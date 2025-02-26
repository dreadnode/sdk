import inspect
import typing as t
from dataclasses import dataclass

from opentelemetry.trace import Tracer

from .tracing import Score, TaskSpan, current_run_span

P = t.ParamSpec("P")
R = t.TypeVar("R")


@dataclass
class Task(t.Generic[P, R]):
    tracer: Tracer

    name: str
    attributes: dict[str, t.Any]
    func: t.Callable[P, t.Awaitable[R]]
    scorers: list["Scorer[R]"]
    tags: t.Sequence[str]

    def __post_init__(self) -> None:
        self.__signature__ = inspect.signature(self.func)
        self.__name__ = getattr(self.func, "__name__", self.name)

    def _bind_args(self, *args: P.args, **kwargs: P.kwargs) -> dict[str, t.Any]:
        signature = inspect.signature(self.func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return dict(bound_args.arguments)

    async def run(self, *args: P.args, **kwargs: P.kwargs) -> TaskSpan[R]:
        run = current_run_span.get()
        if run is None or not run.is_recording:
            raise RuntimeError("Tasks must be executed within a run")

        task = TaskSpan[R](
            name=self.name,
            attributes=self.attributes,
            args=self._bind_args(*args, **kwargs),
            run_id=run.run_id,
            tracer=self.tracer,
        )

        with task:
            output = t.cast(R | t.Awaitable[R], self.func(*args, **kwargs))
            if inspect.isawaitable(output):
                output = await output

            task.output = output

            for scorer in self.scorers:
                score = await scorer(task.output)
                if not isinstance(score, Score):
                    score = Score(name=scorer.name, value=float(score))
                score.name = scorer.name
                task.scores.append(score)

        return task

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        span = await self.run(*args, **kwargs)
        return span.output  # type: ignore


Scorer = Task[P, float | Score]
