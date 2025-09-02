import typing as t
from abc import ABC, abstractmethod

from dreadnode.tracing.span import RunSpan, current_run_span, current_task_span
from dreadnode.types import UNSET, Unset
from dreadnode.util import warn_at_user_stacklevel


class ContextWarning(UserWarning):
    """Warning for issues during context resolution."""


class Context(ABC):
    """
    The abstract base class for all runtime dependency injection markers.

    Subclasses must implement the `resolve` method, which contains the logic
    for retrieving a value by name.
    """

    def __init__(self, *, default: t.Any | Unset = UNSET, required: bool = True):
        self.required = required
        self.default = default
        self._param_name: str | Unset = UNSET

    def __repr__(self) -> str:
        parts = [
            f"name='{self._param_name!r}'",
            f"required={self.required!r}",
        ]
        if self.default is not UNSET:
            parts.append(f"default={self.default!r}")
        return f"Context({', '.join(parts)})"

    @abstractmethod
    def resolve(self) -> t.Any:
        """
        Resolves the dependency's value.

        Returns:
            The resolved value for the dependency.
        """
        raise NotImplementedError


class CurrentRun(Context):
    """
    Retrieve the current run span from the current context.
    """

    def resolve(self) -> t.Any:
        if (run := current_run_span.get()) is None:
            raise RuntimeError("CurrentRun() must be used inside an active run")
        return run


class CurrentTask(Context):
    """
    Retrieve the current task span from the current context.
    """

    def resolve(self) -> t.Any:
        if (task := current_task_span.get()) is None:
            raise RuntimeError("CurrentTask() must be used inside an active task")
        return task


class ParentTask(Context):
    """
    Retrieve the parent of the current task span from the current context.
    """

    def resolve(self) -> t.Any:
        if (task := current_task_span.get()) is None:
            raise RuntimeError("ParentTask() must be used inside an active task")
        if (parent := task.parent) is None:
            raise RuntimeError("ParentTask() must be used inside a nested task")
        return parent


SpanContextSource = t.Literal["input", "output", "param"]
SpanContextScope = t.Literal["task", "run"]


class SpanContext(Context):
    """
    A Context marker for a dynamic value within a Task or Run context.

    This allows scorers and other components to declaratively access inputs, outputs,
    and parameters of the current execution without needing to be explicitly passed them.
    """

    def __init__(
        self,
        name: str,
        source: SpanContextSource,
        *,
        scope: SpanContextScope = "task",
        process: t.Callable[[t.Any], t.Any] | None = None,
        default: t.Any | Unset = UNSET,
        required: bool = True,
    ) -> None:
        """
        Args:
            name: The name of the value to retrieve.
            source: The source to retrieve from ('input', 'output', 'param').
            scope: The scope to look in ('task' or 'run'). Defaults to 'task'.
            process: An optional function to process the retrieved value.
            default: A default value if the named value is not found.
            required: Whether the context is required or not (otherwise use `default` or `None`).
        """
        if source == "param" and scope != "run":
            raise ValueError("Parameters are always run-scoped. Please use scope='run'.")

        super().__init__(default=default, required=required)

        self.ref_name = name
        self.source = source
        self.scope = scope
        self.process = process

    def __repr__(self) -> str:
        parts = [
            f"name='{self.ref_name!r}'",
            f"source='{self.source}'",
            f"scope='{self.scope}'",
            f"required={self.required!r}",
        ]
        if self.default is not UNSET:
            parts.append(f"default={self.default!r}")
        if self.process is not None:
            parts.append(f"process={self.process}")
        return f"SpanContext({', '.join(parts)})"

    def resolve(self) -> t.Any:
        task = current_task_span.get()
        run = current_run_span.get()

        if (target_span := task if self.scope == "task" else run) is None:
            raise RuntimeError(f"No active '{self.scope}' span in context.")

        value_container: t.Any = None
        if self.source == "input":
            value_container = target_span.inputs
        elif self.source == "output":
            value_container = target_span.outputs
        elif self.source == "param":
            if not isinstance(target_span, RunSpan):
                raise RuntimeError("Cannot resolve parameter from non-run scope.")
            value_container = target_span.params

        value: t.Any = None
        try:
            value = value_container[self.ref_name]
        except (KeyError, AttributeError) as e:
            available = list(value_container.keys()) if value_container else []
            raise RuntimeError(
                f"{self.source.capitalize()} '{self.ref_name}' not found in active '{self.scope}' span. "
                f"Available: {available}"
            ) from e

        processed_value = value
        if self.process:
            try:
                processed_value = self.process(value)
            except Exception as e:  # noqa: BLE001
                warn_at_user_stacklevel(f"Error processing {self!r}: {e}", ContextWarning)

        return processed_value


def TaskInput(  # noqa: N802
    name: str, *, default: t.Any | Unset = UNSET, required: bool = True
) -> SpanContext:
    """
    Reference an input from the nearest task.

    Args:
        name: The name of the input to reference (otherwise the parameter name is used).
        default: A default value if the named input is not found.
        required: Whether the context is required or not (otherwise use `default` or `None`).
    """
    return SpanContext(name, "input", scope="task", default=default, required=required)


def TaskOutput(  # noqa: N802
    name: str, *, default: t.Any | Unset = UNSET, required: bool = True
) -> SpanContext:
    """
    Reference an output from the nearest task.

    Args:
        name: The name of the output to reference (otherwise the parameter name is used).
        default: A default value if the named output is not found.
        required: Whether the context is required or not (otherwise use `default` or `None`).
    """
    return SpanContext(name, "output", scope="task", default=default, required=required)


def RunParam(  # noqa: N802
    name: str, *, default: t.Any | Unset = UNSET, required: bool = True
) -> SpanContext:
    """
    Reference a parameter from the current run.

    Args:
        name: The name of the parameter to reference (otherwise the parameter name is used).
        default: A default value if the named parameter is not found.
        required: Whether the context is required or not (otherwise use `default` or `None`).
    """
    return SpanContext(name, "param", scope="run", default=default, required=required)


def RunInput(  # noqa: N802
    name: str, *, default: t.Any | Unset = UNSET, required: bool = True
) -> SpanContext:
    """
    Reference an input from the current run.

    Args:
        name: The name of the input to reference (otherwise the parameter name is used).
        default: A default value if the named input is not found.
        required: Whether the context is required or not (otherwise use `default` or `None`).
    """
    return SpanContext(name, "input", scope="run", default=default, required=required)


def RunOutput(  # noqa: N802
    name: str, *, default: t.Any | Unset = UNSET, required: bool = True
) -> SpanContext:
    """
    Reference an output from the current run.

    Args:
        name: The name of the output to reference (otherwise the parameter name is used).
        default: A default value if the named output is not found.
        required: Whether the context is required or not (otherwise use `default` or `None`).
    """
    return SpanContext(name, "output", scope="run", default=default, required=required)


class DatasetField(Context):
    """
    A Context marker for a value from the full dataset sample row
    for the current evaluation task.
    """

    def __init__(self, name: str, *, default: t.Any | Unset = UNSET, required: bool = True):
        super().__init__(default=default, required=required)
        self.ref_name = name

    def __repr__(self) -> str:
        return f"DatasetField(name='{self.ref_name}')"

    def resolve(self) -> t.Any:
        from dreadnode.evals.evals import current_sample_row

        if (row := current_sample_row.get()) is None:
            raise RuntimeError("DatasetField() can only be used within an active Eval.")
        try:
            return row[self.ref_name]
        except Exception as e:
            available = list(row.keys())
            raise RuntimeError(
                f"Field '{self.ref_name}' not found in dataset sample. "
                f"Available fields: {available}"
            ) from e
