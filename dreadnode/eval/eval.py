import contextlib
import typing as t
from contextlib import asynccontextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict, FilePath, PrivateAttr, TypeAdapter

from dreadnode.configurable import (
    configurable,
)
from dreadnode.discovery import find
from dreadnode.eval.dataset import (
    EvalResult,
    InputDataset,
    InputDatasetProcessor,
    InputT,
    OutputT,
    Sample,
    load_from_file,
)
from dreadnode.scorers.base import Scorer
from dreadnode.task import Task
from dreadnode.types import AnyDict
from dreadnode.util import get_callable_name, shorten_string


@configurable(["name", "task", "dataset", "scorers", "assertions", "label", "concurrency"])
class Eval(BaseModel, t.Generic[InputT, OutputT]):
    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    name: str | None = None
    """The name of the evaluation."""
    description: str = ""
    """A brief description of the eval's purpose."""
    task: Task[[InputT], OutputT] | str
    """The task to evaluate. Can be a Task object or a string representing qualified task name."""
    dataset: InputDataset[InputT] | list[AnyDict] | FilePath
    """The dataset to use for the evaluation. Can be a list of inputs or a file path to load inputs from."""

    preprocessor: InputDatasetProcessor | None = None
    """Optional preprocessor function to transform the dataset before evaluation."""
    scorers: list[Scorer[OutputT]] | None = None
    """Scorers to evaluate the task's output."""
    assertions: list[Scorer[OutputT]] | None = None
    """Assertions to validate the task's output (scores are resolved as truthy)."""
    label: str | None = None
    """Override the name-derived label for logging."""
    concurrency: int | None = None
    """Maximum number of tasks to run in parallel. If None, runs with unlimited concurrency."""

    _label: str = PrivateAttr()

    def __repr__(self) -> str:
        description = shorten_string(self.description or "", 50)

        parts: list[str] = [
            f"name='{self.name}'",
            f"description='{description}'",
            f"task={self.task!r}",
            f"dataset={self.dataset!r}",
        ]

        if self.scorers:
            scorers = ", ".join(get_callable_name(scorer, short=True) for scorer in self.scorers)
            parts.append(f"scorers=[{scorers}]")
        if self.assertions:
            assertions = ", ".join(
                get_callable_name(assertion, short=True) for assertion in self.assertions
            )
            parts.append(f"assertions=[{assertions}]")
        if self.label:
            label = shorten_string(self.label or "", 50)
            parts.append(f"label='{label}'")
        if self.concurrency is not None:
            parts.append(f"concurrency={self.concurrency}")

        return f"{self.__class__.__name__}({', '.join(parts)})"

    @classmethod
    def _generic_types(cls) -> tuple[type[InputT], type[OutputT]]:
        for c in cls.__mro__:
            metadata = getattr(c, "__pydantic_generic_metadata__", {})
            if len(args := (metadata.get("args", ()) or getattr(c, "__args__", ()))) == 2:  # noqa: PLR2004
                return args  # type: ignore[no-any-return]

        return t.Any, t.Any  # type: ignore[return-value]

    async def _prepare(self) -> tuple[Task[[InputT], OutputT], list[AnyDict]]:
        task = find(Task, self.task) if isinstance(self.task, str) else self.task

        dataset = self.dataset
        if isinstance(self.dataset, str | Path):
            dataset = load_from_file(self.dataset)

        input_type, _ = self._generic_types()
        dataset = TypeAdapter(list[input_type]).validate_python(dataset)  # type: ignore[valid-type]

        if self.preprocessor:
            dataset = self.preprocessor(dataset)

        return task, dataset  # type: ignore[return-value]

    @asynccontextmanager
    async def stream(
        self,
    ) -> t.AsyncIterator[
        t.AsyncGenerator[Sample[InputT, OutputT] | EvalResult[InputT, OutputT], None]
    ]:
        from dreadnode import task_span

        task, dataset = await self._prepare()

        assertion_scorers = Scorer.fit_like(self.assertions or [], attributes={"assertion": True})
        extra_scorers = Scorer.fit_like(self.scorers or []) + assertion_scorers
        eval_task = task.with_(scorers=extra_scorers, append=True)
        eval_name = self.name or f"eval - {eval_task.name}"
        eval_label = self.label or f"eval_{eval_task.label}"

        async def sample_gen() -> t.AsyncGenerator[
            Sample[InputT, OutputT] | EvalResult[InputT, OutputT], None
        ]:
            with task_span(eval_name, label=eval_label, tags=["eval"]):
                samples: list[Sample[InputT, OutputT]] = []

                async with eval_task.stream_map(dataset, concurrency=self.concurrency) as stream:
                    async for span in stream:
                        sample = Sample[InputT, OutputT].from_task(span)
                        samples.append(sample)
                        yield sample

                yield EvalResult[InputT, OutputT](name=eval_name, samples=samples)

        async with contextlib.aclosing(sample_gen()) as gen:
            yield gen

    async def run(self) -> EvalResult[InputT, OutputT]:
        """
        Evaluate the task with the given arguments and return a list of Samples.

        Args:
            args: Either a flat list of the first positional argument, or a dict
                    where each key is a parameter name and the value is either a single value
                    or a list of values to map over.
            scorers: A list of scorers to evaluate the task's output.
            name: The name for the evaluation task.
            label: The label for the evaluation task.
            concurrency: The maximum number of tasks to run in parallel.
                            If None, runs with unlimited concurrency.

        Returns:
            A list of Samples containing the evaluation results.
        """
        async with self.stream() as stream:
            async for sample_or_eval in stream:
                if isinstance(sample_or_eval, EvalResult):
                    return sample_or_eval
            raise RuntimeError("Evaluation failed to complete")
