import typing as t

from dreadnode.tracing.span import RunSpan, current_run_span, current_task_span
from dreadnode.util import warn_at_user_stacklevel

CastT = t.TypeVar("CastT")
SourceType = t.Literal["input", "output", "param"]
ScopeType = t.Literal["task", "run"]


class LookupWarning(UserWarning):
    """Warning for issues during reference resolution."""


class Lookup:
    """
    A lazy lookup for a dynamic value within a Task or Run context.

    This allows scorers and other components to declaratively access inputs, outputs,
    and parameters of the current execution without needing to be explicitly passed them.
    """

    def __init__(
        self,
        name: str,
        source: SourceType,
        *,
        scope: ScopeType = "task",
        process: t.Callable[[t.Any], t.Any] | None = None,
    ) -> None:
        """
        Args:
            name: The name of the value to retrieve.
            source: The source to retrieve from ('input', 'output', 'param').
            scope: The scope to look in ('task' or 'run'). Defaults to 'task'.
            process: An optional function to process the retrieved value.
        """
        self.name = name
        self.source = source
        self.scope = scope
        self.process = process

        if self.source == "param" and self.scope != "run":
            raise ValueError("Parameters are always run-scoped. Please use scope='run'.")

    def __repr__(self) -> str:
        return f"Lookup(name='{self.name}', source='{self.source}', scope='{self.scope}')"

    def resolve(self) -> t.Any:
        """
        Resolves the reference from the current context.

        This method navigates the active TaskSpan and RunSpan to find the desired value.
        """
        task = current_task_span.get()
        run = current_run_span.get()

        target_span = task if self.scope == "task" else run

        if target_span is None:
            warn_at_user_stacklevel(
                f"Lookup('{self.name}') cannot be resolved: no active '{self.scope}' span in context.",
                LookupWarning,
            )
            return None

        value_container: t.Any = None
        if self.source == "input":
            value_container = target_span.inputs
        elif self.source == "output":
            value_container = target_span.outputs
        elif self.source == "param":
            if isinstance(target_span, RunSpan):
                value_container = target_span.params
            else:
                warn_at_user_stacklevel(
                    f"Lookup('{self.name}') cannot resolve param from non-run scope.",
                    LookupWarning,
                )
                return None

        raw_value = None
        try:
            # For inputs/outputs, value_container is a dict of ObjectRefs. We need the actual value.
            if self.source in ("input", "output"):
                raw_value = value_container[self.name].value

            # For params, it's just a direct value.
            else:
                raw_value = value_container[self.name]
        except (KeyError, AttributeError):
            available = list(value_container.keys()) if value_container else []
            warn_at_user_stacklevel(
                f"{self.source.capitalize()} Lookup('{self.name}') not found in active '{self.scope}' span. "
                f"Available: {available}",
                LookupWarning,
            )
            return None

        processed_value = raw_value
        if self.process:
            try:
                processed_value = self.process(raw_value)
            except Exception as e:  # noqa: BLE001
                warn_at_user_stacklevel(
                    f"Error processing Lookup('{self.name}'): {e}", LookupWarning
                )

        return processed_value


def lookup_input(
    name: str,
    *,
    scope: ScopeType = "task",
    process: t.Callable[[t.Any], t.Any] | None = None,
) -> Lookup:
    """A convenience factory for creating a Lookup to a task/run input."""
    return Lookup(name, "input", scope=scope, process=process)


def lookup_output(
    name: str,
    *,
    scope: ScopeType = "task",
    process: t.Callable[[t.Any], t.Any] | None = None,
) -> Lookup:
    """A convenience factory for creating a Lookup to a task/run output."""
    return Lookup(name, "output", scope=scope, process=process)


def lookup_param(name: str, *, process: t.Callable[[t.Any], t.Any] | None = None) -> Lookup:
    """A convenience factory for creating a Lookup to a run parameter."""
    return Lookup(name, "param", scope="run", process=process)


def resolve_lookup(value: t.Any) -> t.Any:
    """
    Resolve a value that may be a Lookup or a direct value.

    If the value is a Lookup, it will be resolved to its actual value.
    If it's not a Lookup, it will be returned as-is.
    """
    if isinstance(value, Lookup):
        return value.resolve()
    return value
