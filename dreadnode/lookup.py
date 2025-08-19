import typing as t

from dreadnode import scorers
from dreadnode.tracing.span import RunSpan, current_run_span, current_task_span
from dreadnode.types import UNSET, Unset
from dreadnode.util import warn_at_user_stacklevel

T = t.TypeVar("T")
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
        default: t.Any | Unset = UNSET,
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
        self.default = default

        if self.source == "param" and self.scope != "run":
            raise ValueError("Parameters are always run-scoped. Please use scope='run'.")

    def __repr__(self) -> str:
        return f"Lookup(name='{self.name}', source='{self.source}', scope='{self.scope}')"

    def as_scorer(self, name: str | None = None) -> scorers.Scorer[t.Any]:
        """
        Convert this Lookup into a Scorer that returns the resolved value as a float score.

        This allows the Lookup to be used seamlessly in scoring contexts.
        """

        async def scorer_func(_: t.Any) -> float:  # Note: `data` is ignored here
            # The scorer's value IS the resolved lookup value.
            return float(self.resolve())

        # The scorer's name is derived from the lookup itself.
        return scorers.Scorer.from_callable(scorer_func, name=name or self.name)

    def __gt__(self, value: float) -> scorers.Scorer[T]:
        return scorers.threshold(self.as_scorer(), gt=value)

    def __lt__(self, value: float) -> scorers.Scorer[T]:
        return scorers.threshold(self.as_scorer(), lt=value)

    def __ge__(self, value: float) -> scorers.Scorer[T]:
        return scorers.threshold(self.as_scorer(), gte=value)

    def __le__(self, value: float) -> scorers.Scorer[T]:
        return scorers.threshold(self.as_scorer(), lte=value)

    def __and__(self, other: scorers.Scorer[T]) -> scorers.Scorer[T]:
        return scorers.and_(self.as_scorer(), other)

    def __or__(self, other: scorers.Scorer[T]) -> scorers.Scorer[T]:
        return scorers.or_(self.as_scorer(), other)

    def __invert__(self) -> scorers.Scorer[T]:
        return scorers.not_(self.as_scorer())  # ~ operator

    def __add__(self, other: scorers.Scorer[T]) -> scorers.Scorer[T]:
        return scorers.add(self.as_scorer(), other)

    def __sub__(self, other: scorers.Scorer[T]) -> scorers.Scorer[T]:
        return scorers.add(self.as_scorer(), scorers.scale(other, -1.0))

    def __mul__(self, weight: float) -> scorers.Scorer[T]:
        return scorers.scale(self.as_scorer(), weight)

    def __rmul__(self, weight: float) -> scorers.Scorer[T]:
        return scorers.scale(self.as_scorer(), weight)

    def resolve(self) -> t.Any:  # noqa: PLR0911
        """
        Resolves the reference from the current context.

        This method navigates the active TaskSpan and RunSpan to find the desired value.
        """
        task = current_task_span.get()
        run = current_run_span.get()

        target_span = task if self.scope == "task" else run

        if target_span is None:
            if self.default is UNSET:
                warn_at_user_stacklevel(
                    f"Lookup('{self.name}') cannot be resolved: no active '{self.scope}' span in context.",
                    LookupWarning,
                )
                return None
            return self.default

        value_container: t.Any = None
        if self.source == "input":
            value_container = target_span.inputs
        elif self.source == "output":
            value_container = target_span.outputs
        elif self.source == "param":
            if isinstance(target_span, RunSpan):
                value_container = target_span.params
            elif self.default is UNSET:
                warn_at_user_stacklevel(
                    f"Lookup('{self.name}') cannot resolve param from non-run scope.",
                    LookupWarning,
                )
                return None
            else:
                return self.default

        value: t.Any = None
        try:
            value = value_container[self.name]
        except (KeyError, AttributeError):
            if self.default is UNSET:
                available = list(value_container.keys()) if value_container else []
                warn_at_user_stacklevel(
                    f"{self.source.capitalize()} Lookup('{self.name}') not found in active '{self.scope}' span. "
                    f"Available: {available}",
                    LookupWarning,
                )
                return None
            return self.default

        processed_value = value
        if self.process:
            try:
                processed_value = self.process(value)
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
    default: t.Any | Unset = UNSET,
) -> Lookup:
    """A convenience factory for creating a Lookup to a task/run input."""
    return Lookup(name, "input", scope=scope, process=process, default=default)


def lookup_output(
    name: str = "output",
    *,
    scope: ScopeType = "task",
    process: t.Callable[[t.Any], t.Any] | None = None,
    default: t.Any | Unset = UNSET,
) -> Lookup:
    """A convenience factory for creating a Lookup to a task/run output."""
    return Lookup(name, "output", scope=scope, process=process, default=default)


def lookup_param(
    name: str, *, process: t.Callable[[t.Any], t.Any] | None = None, default: t.Any | Unset = UNSET
) -> Lookup:
    """A convenience factory for creating a Lookup to a run parameter."""
    return Lookup(name, "param", scope="run", process=process, default=default)


def resolve_lookup(value: t.Any) -> t.Any:
    """
    Resolve a value that may be a Lookup or a direct value.

    If the value is a Lookup, it will be resolved to its actual value.
    If it's not a Lookup, it will be returned as-is.
    """
    if isinstance(value, Lookup):
        return value.resolve()
    return value
