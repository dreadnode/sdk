import inspect
import typing as t
from dataclasses import dataclass

from opentelemetry.trace import Tracer

from .tracing import Score, TaskSpan, current_run_span

T = t.TypeVar("T")
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

    def _bind_args(self, *args: P.args, **kwargs: P.kwargs) -> dict[str, t.Any]:
        signature = inspect.signature(self.func)
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        return dict(bound_args.arguments)

    @t.overload
    async def run(self, *args: P.args, raw: t.Literal[False] = False, **kwargs: P.kwargs) -> R:
        ...

    @t.overload
    async def run(self, *args: P.args, raw: t.Literal[True], **kwargs: P.kwargs) -> TaskSpan[R]:
        ...

    async def run(self, *args: P.args, raw: bool = False, **kwargs: P.kwargs) -> R | TaskSpan[R]:
        run = current_run_span.get()
        if run is None or not run.is_recording:
            raise RuntimeError("Tasks must be executed within a run")

        bound = self._bind_args(*args, **kwargs)
        task = TaskSpan[R](
            name=self.name,
            attributes=self.attributes,
            args=bound,
            run_id=run.run_id,
            tracer=self.tracer,
        )

        with task:
            output = t.cast(R | t.Awaitable[R], self.func(*args, **kwargs))
            if inspect.isawaitable(output):
                output = await output

            task.output = output

            for scorer in self.scorers:
                score = await scorer.run(task.output)
                if not isinstance(score, Score):
                    score = Score(name=scorer.name, value=float(score))
                score.name = scorer.name
                task.scores.append(score)

        return task if raw else task.output

    __call__ = run


Scorer = Task[P, float | Score]
