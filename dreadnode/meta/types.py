import inspect
import types
import typing as t
from copy import deepcopy
from dataclasses import dataclass, field

import pydantic
import typing_extensions as te
from annotated_types import SupportsGt
from pydantic import Field
from pydantic._internal._model_construction import ModelMetaclass
from pydantic_core import PydanticUndefined
from typing_extensions import ParamSpec

from dreadnode.meta.context import Context, ContextWarning
from dreadnode.types import UNSET, AnyDict, Unset
from dreadnode.util import warn_at_user_stacklevel

P = ParamSpec("P")
R = t.TypeVar("R")
T = t.TypeVar("T")


@dataclass(frozen=True)
class ConfigInfo:
    """Internal container for static configuration metadata."""

    field_kwargs: dict[str, t.Any] = field(default_factory=dict)
    expose_as: t.Any = None


@t.overload
def Config(
    default: types.EllipsisType,
    *,
    key: str | None = None,
    help: str | None = None,
    description: str | None = None,
    expose_as: t.Any | None = None,
    examples: list[t.Any] | None = None,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    pattern: str | None = None,
    alias: str | None = None,
    **kwargs: t.Any,
) -> t.Any: ...


@t.overload
def Config(
    default: T,
    *,
    key: str | None = None,
    help: str | None = None,
    description: str | None = None,
    expose_as: t.Any = None,
    examples: list[t.Any] | None = None,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    pattern: str | None = None,
    alias: str | None = None,
    **kwargs: t.Any,
) -> T: ...


@t.overload
def Config(
    *,
    default_factory: t.Callable[[], T],
    key: str | None = None,
    help: str | None = None,
    description: str | None = None,
    expose_as: t.Any | None = None,
    examples: list[t.Any] | None = None,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    pattern: str | None = None,
    alias: str | None = None,
    **kwargs: t.Any,
) -> T: ...


@t.overload
def Config(
    *,
    key: str | None = None,
    help: str | None = None,
    description: str | None = None,
    expose_as: t.Any | None = None,
    examples: list[t.Any] | None = None,
    gt: float | None = None,
    ge: float | None = None,
    lt: float | None = None,
    le: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    pattern: str | None = None,
    alias: str | None = None,
    **kwargs: t.Any,
) -> t.Any: ...


def Config(  # noqa: N802
    default: t.Any = ...,
    *,
    key: str | None = UNSET,
    help: str | None = UNSET,
    description: str | None = UNSET,
    expose_as: t.Any | None = None,
    examples: list[t.Any] | None = UNSET,
    exclude: bool | None = UNSET,
    repr: bool = UNSET,
    init: bool | None = UNSET,
    init_var: bool | None = UNSET,
    kw_only: bool | None = UNSET,
    gt: SupportsGt | None = UNSET,
    ge: SupportsGt | None = UNSET,
    lt: SupportsGt | None = UNSET,
    le: SupportsGt | None = UNSET,
    min_length: int | None = UNSET,
    max_length: int | None = UNSET,
    pattern: str | None = UNSET,
    alias: str | None = UNSET,
    **kwargs: t.Any,
) -> t.Any:
    """
    Declares a static, configurable parameter.

    Args:
        default: Default value if the field is not set.
        default_factory: A callable to generate the default value. The callable can either take 0 arguments
            (in which case it is called as is) or a single argument containing the already validated data.
        alias: The name to use for the attribute when validating or serializing by alias.
            This is often used for things like converting between snake and camel case.
        help: Human-readable help text.
        description: Human-readable description (overridden by `help`)
        expose_as: Override the type that this config value should be annotated as in configuration models.
        examples: Example values for this field.
        exclude: Exclude the field from the model serialization.
        repr: A boolean indicating whether to include the field in the `__repr__` output.
        init: Whether the field should be included in the constructor of the dataclass.
            (Only applies to dataclasses.)
        init_var: Whether the field should _only_ be included in the constructor of the dataclass.
            (Only applies to dataclasses.)
        kw_only: Whether the field should be a keyword-only argument in the constructor of the dataclass.
            (Only applies to dataclasses.)
        coerce_numbers_to_str: Enable coercion of any `Number` type to `str` (not applicable in `strict` mode).
        strict: If `True`, strict validation is applied to the field.
            See [Strict Mode](../concepts/strict_mode.md) for details.
        gt: Greater than. If set, value must be greater than this. Only applicable to numbers.
        ge: Greater than or equal. If set, value must be greater than or equal to this. Only applicable to numbers.
        lt: Less than. If set, value must be less than this. Only applicable to numbers.
        le: Less than or equal. If set, value must be less than or equal to this. Only applicable to numbers.
        multiple_of: Value must be a multiple of this. Only applicable to numbers.
        min_length: Minimum length for iterables.
        max_length: Maximum length for iterables.
        pattern: Pattern for strings (a regular expression).
        allow_inf_nan: Allow `inf`, `-inf`, `nan`. Only applicable to float and [`Decimal`][decimal.Decimal] numbers.
        max_digits: Maximum number of allow digits for strings.
        decimal_places: Maximum number of decimal places allowed for numbers.
        union_mode: The strategy to apply when validating a union. Can be `smart` (the default), or `left_to_right`.
            See [Union Mode](../concepts/unions.md#union-modes) for details.
        fail_fast: If `True`, validation will stop on the first error. If `False`, all validation errors will be collected.
            This option can be applied only to iterable types (list, tuple, set, and frozenset).

    """

    if isinstance(default, ConfigInfo):
        return default

    field_kwargs = kwargs
    field_kwargs.update(
        {
            "default": default,
            "description": help or description,  # `help` overrides `description`
            "examples": examples,
            "exclude": exclude,
            "repr": repr,
            "init": init,
            "init_var": init_var,
            "kw_only": kw_only,
            "gt": gt,
            "ge": ge,
            "lt": lt,
            "le": le,
            "min_length": min_length,
            "max_length": max_length,
            "pattern": pattern,
            "alias": key or alias,  # `key` overrides alias
        }
    )

    # Filter UNSET values
    field_kwargs = {k: v for k, v in field_kwargs.items() if v is not UNSET}

    return ConfigInfo(field_kwargs=field_kwargs, expose_as=expose_as)


class ModelMeta(ModelMetaclass):
    def __new__(
        mcs,
        name: str,
        bases: tuple[type[t.Any], ...],
        namespace: dict[str, t.Any],
        **kwargs: t.Any,
    ) -> type:
        for attr_name, attr_value in namespace.items():
            if isinstance(attr_value, ConfigInfo):
                field_kwargs = {
                    k: (v if v is not UNSET else PydanticUndefined)
                    for k, v in attr_value.field_kwargs.items()
                }
                namespace[attr_name] = Field(**field_kwargs)  # type: ignore[arg-type]

        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        params = {
            name: getattr(bases[0] if bases else object, name, attr_value)
            for name, attr_value in namespace.items()
            if isinstance(attr_value, ConfigInfo)
        }
        cls.__dn_config__ = params  # type: ignore[attr-defined]

        return cls


class Model(pydantic.BaseModel, metaclass=ModelMeta):
    pass


# class Model(pydantic.BaseModel):
#     """The base class for all configurable class-based components."""

#     def __init_subclass__(cls, **kwargs: t.Any) -> None:
#         super().__init_subclass__(**kwargs)

#         params: dict[str, ConfigInfo] = {}
#         for name in cls.__annotations__:
#             obj = hasattr(cls, name) and getattr(cls, name)
#             if obj and isinstance(obj, ConfigInfo):
#                 # json_schema_extra = {
#                 #     **(obj.field_kwargs.get("json_schema_extra", {})),
#                 #     "__dn_param__": True,
#                 # }
#                 # obj.field_kwargs["json_schema_extra"] = json_schema_extra
#                 field_kwargs = {
#                     k: (v if v is not UNSET else PydanticUndefined)
#                     for k, v in obj.field_kwargs.items()
#                 }
#                 setattr(cls, name, Field(**field_kwargs))  # type: ignore[arg-type]
#                 params[name] = obj

#         cls.__dn_config__ = params  # type: ignore[attr-defined]


class Component(t.Generic[P, R]):
    """
    A stateful wrapper for a configurable function-based blueprint.
    """

    def __init__(
        self,
        func: t.Callable[P, R],
        *,
        config: dict[str, ConfigInfo] | None = None,
        context: dict[str, Context] | None = None,
    ) -> None:
        self.func = func
        "The underlying function to call"
        self.signature = getattr(func, "__signature__", inspect.signature(func))
        "The underlying function signature"
        self.__dn_param_config__ = config or {
            name: param.default
            for name, param in self.signature.parameters.items()
            if isinstance(param.default, ConfigInfo)
        }
        self.__dn_attr_config__: dict[str, ConfigInfo] = {}
        self.__dn_context__: dict[str, Context] = context or {
            n: p.default
            for n, p in self.signature.parameters.items()
            if isinstance(p.default, Context)
        }
        self.__name__ = func.__name__
        self.__qualname__ = func.__qualname__
        self.__doc__ = func.__doc__

        # Strip any Config values from annotations to avoid
        # them polluting further inspection.
        self.__annotations__ = {
            name: annotation
            for name, annotation in func.__annotations__.items()
            if name not in self.__dn_param_config__
        }
        self.__signature__ = self.signature.replace(
            parameters=[
                param
                for name, param in self.signature.parameters.items()
                if name not in self.__dn_param_config__
            ]
        )

        # Update the parameter names for context dependencies
        for name, dep in self.__dn_context__.items():
            dep._param_name = name  # noqa: SLF001

    # We need this otherwise we could trigger undeseriable behavior
    # when included in deepcopy calls above us
    def __deepcopy__(self, memo: dict[int, t.Any]) -> te.Self:
        return self.__class__(
            self.func,
            config=deepcopy(self.__dn_param_config__, memo),
            context=deepcopy(self.__dn_context__, memo),
        )

    def clone(self) -> te.Self:
        """
        Create a copy of the component with the same configuration and context.
        """
        return self.__deepcopy__({})

    def configure(self, **overrides: t.Any) -> te.Self:
        """
        Configure the component with new default configuration values.

        Keyword arguments are interpreted as any new default values for arguments.

        Examples:
            ```python
            @component
            def my_component(required: int, *, optional: str = Config("default")) -> None:
                pass

            updated = my_component.configure(optional="override")
            ```

        Args:
            **overrides: Any new default values for the component's arguments.

        Returns:
            A new component instance with the updated configuration.
        """
        new = self.clone()

        known_keys = set(new.__dn_param_config__) | set(new.__dn_context__)
        for key, value in overrides.items():
            if key not in known_keys:
                continue

            new.__dn_context__.pop(key, None)
            config = new.__dn_param_config__.pop(key, None)

            if isinstance(value, Context):
                new.__dn_context__[key] = value
                continue

            if isinstance(value, ConfigInfo):
                new.__dn_param_config__[key] = value
                continue

            field_kwargs = {**(config.field_kwargs.copy() if config else {}), "default": value}
            new.__dn_param_config__[key] = ConfigInfo(field_kwargs=field_kwargs)

        return new

    def _bind_args(self, *args: P.args, **kwargs: P.kwargs) -> inspect.BoundArguments:
        """
        Bind the given arguments to the component's signature, resolving configuration and context values."""

        partial_args = self.signature.bind_partial(*args, **kwargs)

        args_dict: AnyDict = {}
        for name in self.signature.parameters:
            if name in partial_args.arguments:
                args_dict[name] = partial_args.arguments[name]
                continue

            if name in self.__dn_param_config__:
                default_value = self.__dn_param_config__[name].field_kwargs.get("default", UNSET)
                default_factory = self.__dn_param_config__[name].field_kwargs.get("default_factory")
                if default_value in (..., PydanticUndefined, UNSET):
                    if default_factory is not None:
                        default_value = default_factory()
                    else:
                        raise TypeError(f"Missing required configuration: '{name}'")
                args_dict[name] = default_value

            if name in self.__dn_context__:
                context = self.__dn_context__[name]
                resolved: t.Any | Unset = UNSET

                try:
                    resolved = context.resolve()
                    if resolved is UNSET and context.required:
                        raise TypeError(f"{context!r} did not resolve to a value")  # noqa: TRY301
                except Exception as e:
                    if (resolved := context.default) is UNSET:
                        if context.required:
                            raise TypeError(f"Missing required dependency: '{name}'") from e
                        resolved = None
                        warn_at_user_stacklevel(f"Failed to resolve '{name}': {e}", ContextWarning)

                args_dict[name] = resolved

        bound_args = self.signature.bind(**args_dict)
        bound_args.apply_defaults()

        return bound_args

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        bound_args = self._bind_args(*args, **kwargs)
        return self.func(*bound_args.args, **bound_args.kwargs)


def component(func: t.Callable[P, R]) -> Component[P, R]:
    return Component(func)
