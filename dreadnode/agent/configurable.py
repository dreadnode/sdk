import contextlib
import functools
import inspect
import types
import typing as t

import jsonref
from pydantic import BaseModel, Field, create_model

from dreadnode.types import AnyDict

if t.TYPE_CHECKING:
    from dreadnode.agent.agent import Agent

CONFIGURABLE_ATTR = "_dn_configurable"
CONFIGURABLE_FIELDS_ATTR = "_dn_configurable_fields"
CONFIGURABLE_FACTORY_ATTR = "_dn_configurable_factory"
CONFIGURABLE_ARGS_ATTR = "_dn_configurable_args"


T = t.TypeVar("T", bound=t.Callable[..., t.Any])


class Configurable:
    """
    A mixin class that marks a component (like an Agent or a Tool) as
    configurable by the CLI.

    By default, it exposes all parameters to the CLI.
    If you want to limit which parameters are exposed, use the `config_fields`
    keyword argument in the class definition.

    Example:
        class MyComponent(Configurable, config_fields=["param1", "param2"]):
            param1: str
            param2: int = 42
            param3: bool = True  # This will not be exposed to the CLI
    """

    _dn_configurable: t.ClassVar[bool] = True
    _dn_configurable_fields: t.ClassVar[list[str] | None] = None

    def __init_subclass__(cls, test: list[str] | None = None, **kwargs: t.Any):
        """
        This method is called when a class inherits from ConfigurableMixin.
        It captures the `fields` keyword argument from the class definition.
        """
        super().__init_subclass__(**kwargs)
        if test is not None:
            cls._dn_configurable_fields = test


def configurable(
    arg: T | list[str] | None = None,
) -> T | t.Callable[[T], T]:
    """
    A decorator to mark a component as configurable by the CLI.

    When decorating a factory function (a function that returns another function),
    it intelligently tags the returned function with metadata, allowing its
    original parameters to be discovered for configuration.

    Can be used in two ways:
    1.  `@configurable`: Exposes all parameters to the CLI.
    2.  `@configurable(["param1"])`: Exposes only specified parameters.
    """

    def decorator(func: T) -> T:
        # Tag the factory function itself so we can identify it.
        setattr(func, CONFIGURABLE_ATTR, True)
        fields = arg if isinstance(arg, list) else None
        setattr(func, CONFIGURABLE_FIELDS_ATTR, fields)

        @functools.wraps(func)
        def factory_wrapper(*args: t.Any, **kwargs: t.Any) -> t.Any:
            # Call the user's factory function to get the final object (e.g., the hook).
            result = func(*args, **kwargs)

            # Tag the result with a reference back to its factory and the args used.
            if callable(result):
                setattr(result, CONFIGURABLE_FACTORY_ATTR, func)
                # Store the arguments that were used to create this specific instance.
                # This is how we get the default values like `max_tokens=1000`.
                bound_args = inspect.signature(func).bind(*args, **kwargs)
                bound_args.apply_defaults()
                setattr(result, CONFIGURABLE_ARGS_ATTR, bound_args.arguments)

            return result

        return t.cast("T", factory_wrapper)

    if callable(arg):
        # Called as `@configurable`
        return decorator(arg)  # type: ignore[arg-type]

    # Called as `@configurable([...])` or `@configurable()`
    return decorator


PRIMITIVE_TYPES = {str, int, bool, float, type(None)}


def _is_cli_friendly_type(annotation: t.Any) -> bool:
    """Checks if a type annotation is a primitive the CLI can handle."""
    origin = t.get_origin(annotation)
    if origin is None:  # It's not a generic like list or Union
        return annotation in PRIMITIVE_TYPES

    if origin in (list, t.Union, types.UnionType):
        return all(arg in PRIMITIVE_TYPES for arg in t.get_args(annotation))

    if origin is dict:
        args = t.get_args(annotation)
        if len(args) != 2:  # noqa: PLR2004
            return False
        key_type, value_type = args
        return key_type in PRIMITIVE_TYPES and value_type in PRIMITIVE_TYPES

    return False


def _make_model(
    name: str, obj: t.Callable[..., t.Any], defaults: AnyDict
) -> type[BaseModel] | None:
    """Dynamically creates a Pydantic BaseModel for a single component."""

    with contextlib.suppress(Exception):
        # Check for the flag from either the Mixin or the decorator
        if not getattr(obj, CONFIGURABLE_ATTR, False):
            return None

        # Get allowed fields from either the Mixin or the decorator
        allowed_fields: list[str] | None = getattr(obj, CONFIGURABLE_FIELDS_ATTR, None)
        model_fields: AnyDict = {}

        # If the object is already a Pydantic model, use its fields directly
        if issubclass(obj, BaseModel):  # type: ignore[arg-type]
            for field_name, field in obj.model_fields.items():  # type: ignore[attr-defined]
                if allowed_fields is None or field_name in allowed_fields:
                    model_fields[field_name] = (field.annotation, field)

            if not model_fields:
                return None

            return create_model(name, **model_fields)

        # Otherwise use the signature to extract fields
        @functools.wraps(obj)
        def empty_func(*args, **kwargs):  # type: ignore [no-untyped-def] # noqa: ARG001
            return kwargs

        # Clear the return annotation to help reduce errors
        empty_func.__annotations__.pop("return", None)

        # Clear any arguments not in allowed_fields
        if allowed_fields is not None:
            empty_func.__annotations__ = {
                k: v for k, v in empty_func.__annotations__.items() if k in allowed_fields
            }

        try:
            signature = inspect.signature(empty_func, eval_str=True)
        except (ValueError, TypeError):
            return None

        for param in signature.parameters.values():
            if param.name == "self" or not _is_cli_friendly_type(param.annotation):
                continue

            default_value = defaults.get(param.name, param.default)
            instance = defaults.get("__instance__")
            if not callable(instance) and hasattr(instance, param.name):
                default_value = getattr(instance, param.name)

            model_fields[param.name] = (
                param.annotation,
                Field(default=... if default_value is inspect.Parameter.empty else default_value),
            )

        if not model_fields:
            return None

        return create_model(name, **model_fields)

    return None


def generate_config_model(agent: "Agent") -> type[BaseModel]:
    top_level_fields: AnyDict = {}

    # Agent model generation remains the same, assuming Agent inherits from ConfigurableMixin
    if isinstance(agent, Configurable) and (
        agent_model := _make_model("agent", type(agent), {"__instance__": agent}),
    ):
        top_level_fields["agent"] = (agent_model, Field(default=...))

    # Process tools and hooks
    components: dict[str, list[t.Any]] = {"tools": agent.tools, "hooks": agent.hooks}
    for group_name, component_list in components.items():
        group_fields: AnyDict = {}
        for component in component_list:
            target_obj = None
            defaults: AnyDict = {}

            if hasattr(component, CONFIGURABLE_FACTORY_ATTR):
                target_obj = getattr(component, CONFIGURABLE_FACTORY_ATTR)
                defaults = getattr(component, CONFIGURABLE_ARGS_ATTR, {})
            elif isinstance(component, Configurable):
                target_obj = type(component)
                defaults = {"__instance__": component}

            if not target_obj:
                continue

            if component_model := _make_model(target_obj.__name__, target_obj, defaults):
                group_fields[target_obj.__name__] = (component_model, Field(default=...))

        if group_fields:
            top_level_fields[group_name] = (
                create_model(group_name, **group_fields),
                Field(default=...),
            )

    if not top_level_fields:
        return {"type": "object", "properties": {}}

    return create_model(agent.name, **top_level_fields)


def generate_config_schema(agent: "Agent") -> AnyDict:
    model = generate_config_model(agent)
    schema = model.model_json_schema()
    schema = t.cast("AnyDict", jsonref.replace_refs(schema, proxies=False, lazy_load=False))
    schema.pop("$defs", None)  # Remove $defs if present
    return schema
