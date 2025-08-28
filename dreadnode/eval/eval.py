import contextlib
import typing as t
from contextlib import asynccontextmanager
from pathlib import Path

from pydantic import ConfigDict, FilePath, TypeAdapter

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
from dreadnode.meta import Model
from dreadnode.meta.types import Config
from dreadnode.scorers.base import Scorer, ScorersLike
from dreadnode.task import Task
from dreadnode.types import AnyDict
from dreadnode.util import get_callable_name, shorten_string


class Eval(Model, t.Generic[InputT, OutputT]):
    model_config = ConfigDict(arbitrary_types_allowed=True, use_attribute_docstrings=True)

    name: str | None = Config(None)
    """The name of the evaluation."""
    description: str = Config("")
    """A brief description of the eval's purpose."""
    task: Task[[InputT], OutputT] | str = Config(expose_as=str)
    """The task to evaluate. Can be a Task object or a string representing qualified task name."""
    dataset: InputDataset[InputT] | list[AnyDict] | FilePath = Config(expose_as=FilePath)
    """The dataset to use for the evaluation. Can be a list of inputs or a file path to load inputs from."""
    concurrency: int = Config(1)
    """Maximum number of tasks to run in parallel."""

    preprocessor: InputDatasetProcessor | None = None
    """Optional preprocessor function to transform the dataset before evaluation."""
    scorers: ScorersLike[OutputT] = Config(default_factory=list)
    """Scorers to evaluate the task's output."""
    assertions: ScorersLike[OutputT] = Config(default_factory=list)
    """Assertions to validate the task's output (scores are resolved as truthy)."""

    def __repr__(self) -> str:
        description = shorten_string(self.description or "", 50)

        parts: list[str] = [
            f"name='{self.name}'",
            f"description='{description}'",
            f"task={self.task!r}",
            f"dataset={self.dataset!r}",
        ]

        if self.scorers:
            scorers = ", ".join(
                get_callable_name(scorer, short=True) for scorer in Scorer.fit_like(self.scorers)
            )
            parts.append(f"scorers=[{scorers}]")
        if self.assertions:
            assertions = ", ".join(
                get_callable_name(assertion, short=True)
                for assertion in Scorer.fit_like(self.assertions)
            )
            parts.append(f"assertions=[{assertions}]")
        if self.concurrency is not None:
            parts.append(f"concurrency={self.concurrency}")

        return f"{self.__class__.__name__}({', '.join(parts)})"

    @classmethod
    def _generic_types(cls) -> tuple[type[InputT], type[OutputT]]:
        """
        Extract the generic types (InputT, OutputT) from the class hierarchy.

        This method traverses the Method Resolution Order (MRO) to find the first class
        that has generic type arguments defined, either in Pydantic generic metadata
        or in the class's __args__ attribute. This is used for type validation and
        ensuring proper type safety throughout the evaluation framework.

        Returns:
            A tuple containing the input type and output type. If no generic types
            are found in the class hierarchy, returns (Any, Any) as fallback types.

        Example:
            For a class like Eval[str, int], this would return (str, int).
        """
        for c in cls.__mro__:
            metadata = getattr(c, "__pydantic_generic_metadata__", {})
            if len(args := (metadata.get("args", ()) or getattr(c, "__args__", ()))) == 2:  # noqa: PLR2004
                return args  # type: ignore[no-any-return]

        return t.Any, t.Any  # type: ignore[return-value]

    async def _prepare(self) -> tuple[Task[[InputT], OutputT], list[AnyDict]]:
        """
        Prepare the task and dataset for evaluation by resolving and validating components.

        This method performs several preprocessing steps:
        1. Resolves the task if provided as a string reference
        2. Loads the dataset from file if provided as a file path
        3. Validates the dataset against the expected input type using type adapters
        4. Applies optional preprocessing transformations to the dataset

        The preparation ensures that both the task and dataset are properly typed
        and validated before the evaluation begins, preventing runtime type errors
        and ensuring data consistency.

        Returns:
            A tuple containing:
            - The resolved and validated Task object
            - The processed dataset as a list of dictionaries

        Raises:
            ValidationError: If the dataset doesn't match the expected input type
            ValueError: If the task string reference cannot be resolved
        """
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
        """
        Create an async context manager for streaming evaluation results.

        This method provides a streaming interface for running evaluations, yielding
        individual Sample objects as they complete, followed by a final EvalResult.
        The streaming approach allows for real-time processing and monitoring of
        evaluation progress, especially useful for long-running evaluations.

        The method handles:
        - Task and dataset preparation via _prepare()
        - Configuration of scorers and assertions
        - Concurrent execution of tasks with optional concurrency limits
        - Proper resource cleanup through async context management
        - Telemetry and span tracking for observability

        Yields:
            An async generator that yields:
            - Sample[InputT, OutputT]: Individual evaluation samples as they complete
            - EvalResult[InputT, OutputT]: Final aggregated result containing all samples

        Example:
            ```python
            async with eval_instance.stream() as stream:
                async for item in stream:
                    if isinstance(item, Sample):
                        print(f"Completed sample: {item}")
                    elif isinstance(item, EvalResult):
                        print(f"Final result: {item}")
            ```

        Note:
            The context manager ensures proper cleanup of async resources and
            maintains consistent telemetry spans for the entire evaluation process.
        """
        from dreadnode import task_span

        task, dataset = await self._prepare()

        assertion_scorers = Scorer.fit_like(self.assertions or [], attributes={"assertion": True})
        extra_scorers = Scorer.fit_like(self.scorers or []) + assertion_scorers
        eval_task = task.with_(scorers=extra_scorers, append=True)
        eval_name = self.name or f"eval - {eval_task.name}"

        async def sample_gen() -> t.AsyncGenerator[
            Sample[InputT, OutputT] | EvalResult[InputT, OutputT], None
        ]:
            with task_span(eval_name, tags=["eval"]):
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
        """
        async with self.stream() as stream:
            async for sample_or_eval in stream:
                if isinstance(sample_or_eval, EvalResult):
                    return sample_or_eval
            raise RuntimeError("Evaluation failed to complete")
