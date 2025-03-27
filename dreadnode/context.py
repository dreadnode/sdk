import typing as t
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Annotated, Any, Generic, TypeVar, get_args, get_origin

from fast_depends.library import CustomField

T = TypeVar("T")


class ContextField(CustomField, Generic[T]):
    """A custom field that resolves values from the current run context."""

    def __init__(
        self,
        id_: str | None = None,
        strict: bool = True,
        default: Any = None,
    ) -> None:
        self.id_ = id_
        self.default = default
        super().__init__(cast=False, required=strict and default is None)

    def use(self, **kwargs: Any) -> dict[str, Any]:
        """Resolve the value from current context when parameter is needed."""
        if self.param_name in kwargs:
            return kwargs

        if (ctx := current_run_context.get()) is None:
            if not self.required:
                kwargs[self.param_name] = self.default
                return kwargs
            raise RuntimeError(f"No active run context for parameter {self.param_name}")

        # Extract the expected type from the parameter annotation
        param_type = self._get_parameter_type()
        if param_type is None:
            raise RuntimeError(f"Could not determine type for parameter {self.param_name}")

        try:
            value = ctx.get(param_type, self.id_, strict=self.required)
            if value is None and self.default is not None:
                value = self.default
            kwargs[self.param_name] = value
        except (KeyError, ValueError) as e:
            if not self.required:
                kwargs[self.param_name] = self.default
            else:
                raise RuntimeError(f"Context resolution failed for {self.param_name}: {e!s}")

        return kwargs

    def _get_parameter_type(self) -> type | None:
        """Extract the actual type from the parameter's type annotation."""
        from fast_depends.core import CallModel

        if not hasattr(self, "_call_model"):
            return None

        model: CallModel = self._call_model
        param_info = model.params.get(self.param_name)
        if param_info is None:
            return None

        annotation = param_info[0]
        if get_origin(annotation) is Annotated:
            # Get the real type from the Annotated type
            return get_args(annotation)[0]
        return annotation


@dataclass
class RunContext:
    """Context manager for a specific run instance."""

    run_id: str
    _global_context: dict[tuple[type, str | None], t.Any] = field(default_factory=dict)
    _scoped_context: dict[tuple[type, str | None], t.Any] = field(default_factory=dict)

    def set_global(self, value: t.Any, id_: str | None = None) -> None:
        """Set a global context value for this run."""
        self._global_context[type(value), id_] = value

    def set_scoped(self, value: t.Any, id_: str | None = None) -> None:
        """Set a scoped context value."""
        self._scoped_context[type(value), id_] = value

    @t.overload
    def get(self, type_: type[T], id_: str | None = None, strict: t.Literal[True] = True) -> T: ...

    @t.overload
    def get(
        self, type_: type[T], id_: str | None = None, strict: t.Literal[False] = ...
    ) -> T | None: ...

    def get(
        self, type_: type[T], id_: str | list[str] | None = None, strict: bool = True
    ) -> T | None:
        return self._get(type_, id_, strict)

    def _get(self, type_: type[T], id_: str | None = None, strict: bool = True) -> T | None:
        key = (type_, id_)
        if id_ is None:
            # If no ID specified, look for single instance of type
            matching = [(k, v) for k, v in self._scoped_context.items() if issubclass(k[0], type_)]
            if len(matching) == 1:
                return matching[0][1]
            if len(matching) > 1 and strict:
                raise ValueError(
                    f"Multiple {type_.__name__} objects exist. Use a specific id to get one of the values.",
                )
        elif key in self._scoped_context:
            return self._scoped_context[key]

        # Then check global context
        if id_ is None:
            matching = [(k, v) for k, v in self._global_context.items() if issubclass(k[0], type_)]
            if len(matching) == 1:
                return matching[0][1]
            if len(matching) > 1 and strict:
                raise ValueError(
                    f"Multiple {type_.__name__} objects exist. Use a specific id to get one of the values.",
                )
        elif key in self._global_context:
            return self._global_context[key]

        if strict:
            raise KeyError(f"{type_.__name__}{' with id ' + repr(id_) if id_ else ''}")
        return None

    def scope(self, **values: t.Any):
        """Create a scoped context manager."""
        return ScopedContext(self, values)


class ScopedContext:
    """Context manager for temporary scoped values."""

    def __init__(self, context: RunContext, values: dict[str, Any]):
        self.context = context
        self.values = values
        self.previous: dict[tuple[type, str | None], Any] = {}

    def __enter__(self) -> None:
        # Save current values and set new ones
        for id_, value in self.values.items():
            key = (type(value), id_)
            self.previous[key] = self.context._scoped_context.get(key)
            self.context.set_scoped(value, id_)

    def __exit__(self, *exc: object) -> None:
        # Restore previous values
        for key, value in self.previous.items():
            if value is None:
                self.context._scoped_context.pop(key, None)
            else:
                self.context._scoped_context[key] = value


current_run_context: ContextVar[RunContext | None] = ContextVar("current_run_context", default=None)


def get_context() -> RunContext:
    """Get the currently active run context."""
    if (ctx := current_run_context.get()) is None:
        raise RuntimeError("No active run context")
    return ctx


# Context field creation helpers
def ctx(*, id_: str | None = None) -> Any:
    """Create a context field that resolves using the parameter's type annotation."""
    return ContextField(id_=id_)


def optional_ctx(*, id_: str | None = None, default: Any = None) -> Any:
    """Create an optional context field."""
    return ContextField(id_=id_, strict=False, default=default)
