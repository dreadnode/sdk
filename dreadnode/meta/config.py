import contextlib
import inspect
import types
import typing as t
from copy import deepcopy
from dataclasses import dataclass, field
from typing import get_origin

import typing_extensions as te
from annotated_types import SupportsGt
from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field as PydanticField
from pydantic import PrivateAttr as PydanticPrivateAttr
from pydantic import PydanticInvalidForJsonSchema, PydanticSchemaGenerationError, TypeAdapter
from pydantic._internal._model_construction import ModelMetaclass
from pydantic_core import PydanticUndefined
from typing_extensions import ParamSpec

from dreadnode.common_types import UNSET, AnyDict, Unset
from dreadnode.meta.context import Context
from dreadnode.util import clean_str, get_callable_name, safe_issubclass, warn_at_user_stacklevel

P = ParamSpec("P")
R = t.TypeVar("R")
T = t.TypeVar("T")


class ConfigWarning(UserWarning):
    """Warning related to object configurations."""


@dataclass(frozen=True)
class ConfigInfo:
    """Internal container for static configuration metadata."""

    field_kwargs: dict[str, t.Any] = field(default_factory=dict)
    expose_as: t.Any = None

    @staticmethod
    def from_annotation(annotation: t.Any) -> "ConfigInfo | None":
        """Extract ConfigInfo from Annotated metadata."""
        if get_origin(annotation) is t.Annotated:
            args = t.get_args(annotation)
            expose_as = args[0]
            for metadata in args[1:]:
                if isinstance(metadata, ConfigInfo):
                    if metadata.expose_as is None:
                        return ConfigInfo(field_kwargs=metadata.field_kwargs, expose_as=expose_as)
                    return metadata
        return None

    @staticmethod
    def from_defaults_and_annotations(  # noqa: PLR0912
        defaults: AnyDict, annotations: AnyDict, *, convert_all: bool = False
    ) -> dict[str, "ConfigInfo"]:
        """
        Extract ConfigInfo from default values and associated annotations.

        Args:
            defaults: A mapping of parameter names to their default values.
            annotations: A mapping of parameter names to their type annotations.
            convert_all: If True, treat all parameters as potentially configurable,
                even if they lack explicit ConfigInfo.
        """
        configs: dict[str, ConfigInfo] = {}

        configs_from_defaults: dict[str, ConfigInfo] = {}
        for name, value in defaults.items():
            if name == "return":
                continue
            if isinstance(value, ConfigInfo):
                configs_from_defaults[name] = value

            # TODO(nick): Maybe add a logging statement or warning if we can't
            # deal with a particular type during auto conversion (most typically
            # an entrypoint task with complex args)
            elif convert_all and is_type_usable_for_configuration(type(value)):
                configs_from_defaults[name] = ConfigInfo(field_kwargs={"default": value})

        configs_from_annotations: dict[str, ConfigInfo] = {}
        for name, annotation in annotations.items():
            if name == "return":
                continue
            if config := ConfigInfo.from_annotation(annotation):
                configs_from_annotations[name] = config
            elif convert_all and is_type_usable_for_configuration(annotation):
                configs_from_annotations[name] = ConfigInfo()

        for name in set(configs_from_defaults) | set(configs_from_annotations):
            config_from_default = configs_from_defaults.get(name)
            config_from_annotation = configs_from_annotations.get(name)

            # Merge configs if both are present (arg: Annotated[int, Config()] = Config(123))
            if config_from_default and config_from_annotation:
                configs[name] = config_from_annotation.merge(config_from_default)

            # Take from default if available (arg: int = Config())
            elif config_from_default:
                configs[name] = config_from_default

            # Merge default and annotation (arg: Annotated[int, Config()] = 123)
            elif config_from_annotation and name in defaults:
                configs[name] = ConfigInfo(
                    field_kwargs={
                        **config_from_annotation.field_kwargs,
                        "default": defaults[name],
                    },
                    expose_as=config_from_annotation.expose_as,
                )

            # Otherwise just annotation (arg: Annotated[int, Config()])
            elif config_from_annotation:
                configs[name] = config_from_annotation

        return configs

    def merge(self: "ConfigInfo", other: "ConfigInfo") -> "ConfigInfo":
        """Merge configs - `other` takes precedence over `self`."""
        merged_kwargs = {**self.field_kwargs, **other.field_kwargs}
        merged_expose_as = other.expose_as or self.expose_as
        return ConfigInfo(field_kwargs=merged_kwargs, expose_as=merged_expose_as)


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

    if isinstance(default, ConfigInfo | Context):
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


@te.dataclass_transform(
    kw_only_default=True, field_specifiers=(Config, PydanticField, PydanticPrivateAttr)
)
class ConfigurableMeta(ModelMetaclass):
    def __new__(
        cls,
        name: str,
        bases: tuple[type[t.Any], ...],
        namespace: dict[str, t.Any],
        **kwargs: t.Any,
    ) -> type:
        configs = ConfigInfo.from_defaults_and_annotations(
            namespace, namespace.get("__annotations__", {})
        )

        # Rewrite all our configs as pydantic fields
        for attr_name, config in configs.items():
            field_kwargs = {
                k: (v if v is not UNSET else PydanticUndefined)
                for k, v in config.field_kwargs.items()
            }
            namespace[attr_name] = PydanticField(**field_kwargs)  # type: ignore[arg-type]

        new_cls = super().__new__(cls, name, bases, namespace, **kwargs)

        # Merge config from all base classes
        merged_configs = {}
        for base in reversed(new_cls.__mro__):  # Go from most base to most derived
            if hasattr(base, "__dn_config__"):
                merged_configs.update(base.__dn_config__)

        merged_configs.update(configs)

        # If pydantic resolved any of our field descriptions, we need to
        # reflect those back into the ConfigInfo objects
        for field_name, field_info in new_cls.model_fields.items():  # type: ignore[attr-defined]
            if field_name in configs:
                configs[field_name].field_kwargs["description"] = field_info.description

        new_cls.__dn_config__ = merged_configs  # type: ignore[attr-defined]

        return new_cls


class Model(PydanticBaseModel, metaclass=ConfigurableMeta):
    def configure(self, **overrides: t.Any) -> te.Self:
        """Create a new model with updated default configuration values."""
        return self.model_copy(update=overrides)

        # Update the ConfigInfo defaults to match
        # updated_config = {}
        # for name, config_info in t.cast("dict[str, ConfigInfo]", self.__dn_config__).items():
        #     if name in overrides:
        #         new_field_kwargs = {**config_info.field_kwargs, "default": overrides[name]}
        #         updated_config[name] = ConfigInfo(
        #             field_kwargs=new_field_kwargs, expose_as=config_info.expose_as
        #         )
        #     else:
        #         updated_config[name] = config_info

        # new_instance.__dn_config__ = updated_config
        # return new_instance


class Component(t.Generic[P, R]):
    """
    A stateful wrapper for a configurable function-based blueprint.
    """

    def __init__(
        self,
        func: t.Callable[P, R],
        *,
        name: str | None = None,
        config: dict[str, ConfigInfo] | None = None,
        convert_all: bool = False,
        context: dict[str, Context] | None = None,
        wraps: t.Callable[..., t.Any] | None = None,
    ) -> None:
        if name is None:
            unwrapped = inspect.unwrap(wraps or func)
            name = get_callable_name(unwrapped, short=True)

        self.name = clean_str(name)
        "The name of the component."
        self.func = func
        "The underlying function to call"
        self.signature = getattr(wraps or func, "__signature__", inspect.signature(func))
        "The underlying function signature"
        self.__dn_param_config__: dict[str, ConfigInfo] = config or (
            wraps.__dn_param_config__
            if isinstance(wraps, Component)
            else ConfigInfo.from_defaults_and_annotations(
                {
                    n: p.default
                    for n, p in self.signature.parameters.items()
                    if p.default is not inspect.Parameter.empty
                },
                func.__annotations__,
                convert_all=convert_all,
            )
        )
        self.__dn_context__: dict[str, Context] = context or (
            wraps.__dn_context__
            if isinstance(wraps, Component)
            else {
                n: p.default
                for n, p in self.signature.parameters.items()
                if isinstance(p.default, Context)
            }
        )
        self.__name__ = self.name
        self.__qualname__ = (wraps or func).__qualname__
        self.__doc__ = (wraps or func).__doc__

        # Strip any Config values from annotations to avoid
        # them polluting further inspection.
        self.__annotations__ = {
            name: annotation
            for name, annotation in (wraps or func).__annotations__.items()
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

    def __repr__(self) -> str:
        params = ", ".join(
            f"{name}={config.field_kwargs.get('default', '...')!r}"
            for name, config in self.__dn_param_config__.items()
        )
        context = ", ".join(self.__dn_context__.keys())
        if context:
            if params:
                params += ", "
            params += f"*[{context}]"
        return f"{self.__name__}({params})"

    # We need this otherwise we could trigger undeseriable behavior
    # when included in deepcopy calls above us
    def __deepcopy__(self, memo: dict[int, t.Any]) -> te.Self:
        return self.__class__(
            self.func,
            config=deepcopy(self.__dn_param_config__, memo),
            context=deepcopy(self.__dn_context__, memo),
        )

    def clone(self) -> te.Self:
        """Clone the component."""
        return self.__deepcopy__({})

    @property
    def defaults(self) -> dict[str, Unset | t.Any]:
        defaults: dict[str, Unset | t.Any] = {}
        for name, param in self.signature.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue

            if name in self.__dn_param_config__:
                config_info = self.__dn_param_config__[name]
                if "default" in config_info.field_kwargs:
                    defaults[name] = config_info.field_kwargs["default"]
                elif "default_factory" in config_info.field_kwargs:
                    defaults[name] = config_info.field_kwargs["default_factory"]
            elif name in self.__dn_context__:
                defaults[name] = self.__dn_context__[name]
            else:
                defaults[name] = (
                    param.default if param.default is not inspect.Parameter.empty else UNSET
                )
        return defaults

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

        known_keys = (
            set(new.__dn_param_config__) | set(new.__dn_context__) | set(self.signature.parameters)
        )
        for key, value in overrides.items():
            if key not in known_keys:
                warn_at_user_stacklevel(
                    f"Unknown parameter '{key}' passed to {self.__name__}.configure()",
                    ConfigWarning,
                )
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
                args_dict[name] = context.resolve()

        bound_args = self.signature.bind(**args_dict)
        bound_args.apply_defaults()

        return bound_args

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        bound_args = self._bind_args(*args, **kwargs)
        return self.func(*bound_args.args, **bound_args.kwargs)


def component(func: t.Callable[P, R]) -> Component[P, R]:
    return Component(func)


# Utils


def is_type_usable_for_configuration(obj_type: type[t.Any]) -> bool:
    """
    Determines if an type appears valid to be used as part of a config model.

    Args:
        type: The type to check.
    """
    if safe_issubclass(obj_type, (Model, Component)):
        return True

    # This might create some overhead, but it's the safest way
    # to see if we'll run into issues later in our model
    # construction. This could also maybe be inlined into
    # our introspection code, but I'd prefer to avoid creating
    # any ConfigInfo objects that later turn out to be invalid.

    with contextlib.suppress(PydanticInvalidForJsonSchema, PydanticSchemaGenerationError):
        TypeAdapter(obj_type).json_schema()
        return True
    return True
