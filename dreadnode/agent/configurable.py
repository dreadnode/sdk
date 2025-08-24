import functools
import inspect
import types
import typing as t
from copy import deepcopy

import jsonref  # type: ignore[import-untyped]
from loguru import logger
from pydantic import BaseModel, Field, create_model

from dreadnode.types import AnyDict

if t.TYPE_CHECKING:
    from dreadnode.agent.agent import Agent

TypeT = t.TypeVar("TypeT", bound=type)
CallableT = t.TypeVar("CallableT", bound=t.Callable[..., t.Any])
ItemT = t.TypeVar("ItemT", bound=t.Callable[..., t.Any] | type)

CONFIGURABLE_ATTR = "_configurable"
CONFIGURABLE_FIELDS_ATTR = "_configurable_fields"
CONFIGURABLE_FACTORY_ATTR = "_configurable_factory"
CONFIGURABLE_ARGS_ATTR = "_configurable_args"


@t.overload
def configurable(obj: ItemT) -> ItemT: ...


@t.overload
def configurable(obj: list[str] | None = None) -> t.Callable[[ItemT], ItemT]: ...


def configurable(obj: ItemT | list[str] | None = None) -> ItemT | t.Callable[[ItemT], ItemT]:
    """
    A universal decorator to mark a class or a factory function as configurable.

    It can be used in several ways:

    1.  On a class to expose all of its CLI-friendly attributes:
        @configurable
        class MyAgent(BaseModel):
            param1: str

    2.  On a class to expose a specific subset of attributes:
        @configurable(fields=["param1"])
        class MyAgent(BaseModel):
            param1: str
            param2: int # This will not be exposed

    3.  On a factory function to expose all its parameters:
        @configurable
        def my_tool_factory(arg1: int) -> Tool:
            ...

    4.  On a factory function to expose a subset of its parameters:
        @configurable(fields=["arg1"])
        def my_tool_factory(arg1: int, arg2: bool) -> Tool:
            ...
    """

    exposed_fields: list[str] | None = None
    if isinstance(obj, list):
        exposed_fields = obj

    def decorator(obj: ItemT) -> ItemT:
        # Tag the object with the primary configurable markers.
        setattr(obj, CONFIGURABLE_ATTR, True)
        setattr(obj, CONFIGURABLE_FIELDS_ATTR, exposed_fields)

        # If the decorated object is a class, our work is done. Just tag and return.
        if inspect.isclass(obj):
            return obj

        if not callable(obj):
            raise TypeError(
                f"The @configurable decorator can only be applied to classes or functions, "
                f"not to objects of type {type(obj).__name__}."
            )

        # If the decorated object is a function, wrap it to capture factory arguments.
        @functools.wraps(obj)
        def factory_wrapper(*w_args: t.Any, **w_kwargs: t.Any) -> t.Any:
            # Call the user's factory function to get the final object.
            result = obj(*w_args, **w_kwargs)

            # Tag the *result* with a reference back to its factory and the args used.
            # This allows us to inspect its original configuration.
            if callable(result) or hasattr(result, "__class__"):
                setattr(result, CONFIGURABLE_FACTORY_ATTR, obj)
                # Bind the arguments to the factory's signature to get a full
                # picture of the configuration, including default values.
                bound_args = inspect.signature(obj).bind(*w_args, **w_kwargs)
                bound_args.apply_defaults()
                setattr(result, CONFIGURABLE_ARGS_ATTR, bound_args.arguments)

            return result

        return t.cast("ItemT", factory_wrapper)

    if callable(obj) and not isinstance(obj, list):
        return decorator(obj)

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
        if len(args) != 2:
            return False
        key_type, value_type = args
        return key_type in PRIMITIVE_TYPES and value_type in PRIMITIVE_TYPES

    return False


def _safe_issubclass(cls: t.Any, class_or_tuple: TypeT) -> t.TypeGuard[TypeT]:
    """Safely check if a class is a subclass of another class or tuple."""
    try:
        return isinstance(cls, type) and issubclass(cls, class_or_tuple)
    except TypeError:
        return False


def _make_model_fields(obj: t.Callable[..., t.Any], defaults: AnyDict) -> dict[str, t.Any] | None:
    # with contextlib.suppress(Exception):
    # Check for the flag from either the Mixin or the decorator
    if not getattr(obj, CONFIGURABLE_ATTR, False):
        logger.debug(f"{obj.__name__} is not configurable. Skipping.")
        return None

    # Get allowed fields from either the Mixin or the decorator
    allowed_fields: list[str] | None = getattr(obj, CONFIGURABLE_FIELDS_ATTR, None)
    model_fields: AnyDict = {}

    # If the object is already a Pydantic model, use its fields directly
    if _safe_issubclass(obj, BaseModel):
        instance = defaults.get("__instance__")
        for field_name, field in obj.model_fields.items():
            if allowed_fields is not None and field_name not in allowed_fields:
                continue

            if not callable(instance) and hasattr(instance, field_name):
                field.default = getattr(instance, field_name)

            model_fields[field_name] = (field.annotation, field)
        return model_fields

    # Otherwise use the signature to extract fields
    @functools.wraps(obj)
    def empty_func(*args, **kwargs):  # type: ignore [no-untyped-def]
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
    except (ValueError, TypeError, NameError):
        # print(f"Failed to inspect {obj.__name__}: {e}")
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

    return model_fields

    return None


def _make_model(
    name: str, obj: t.Callable[..., t.Any], defaults: AnyDict
) -> type[BaseModel] | None:
    if model_fields := _make_model_fields(obj, defaults):
        return create_model(name, **model_fields)
    return None


def generate_config_type_for_agent(agent: "Agent") -> type[BaseModel]:
    # 1. Top level is the core agent config
    top_level_fields: AnyDict | None = _make_model_fields(type(agent), {"__instance__": agent})
    if top_level_fields is None:
        raise TypeError(
            f"Agent {agent.name} is not configurable. "
            "Ensure it is decorated with @configurable or inherits from ConfigurableMixin."
        )

    # 2. Process Tools and Hooks
    components: dict[str, list[t.Any]] = {"tools": agent.tools, "hooks": agent.hooks}
    for group_name, component_list in components.items():
        group_fields: AnyDict = {}
        for component in component_list:
            target_obj: t.Callable[..., t.Any] | type[BaseModel] | None = None
            defaults: AnyDict = {}

            # A - Component was created by a @configurable factory function
            if factory := getattr(component, CONFIGURABLE_FACTORY_ATTR, None):
                target_obj = factory
                defaults = getattr(component, CONFIGURABLE_ARGS_ATTR, {})

            # B - Component is an instance of a @configurable class
            elif getattr(type(component), CONFIGURABLE_ATTR, False):
                target_obj = type(component)
                defaults = {"__instance__": component}

            if not target_obj:
                continue

            # Use the component's original name for the model key
            component_name = str(getattr(target_obj, "__name__", "unnamed"))
            # component_name = re.sub(r"tool$", "", clean_str(component_name))
            if component_model := _make_model(component_name, target_obj, defaults):
                group_fields[component_name] = (component_model, Field(default=...))

        if group_fields:
            group_model = create_model(group_name, **group_fields)
            top_level_fields[group_name] = (group_model, Field(default={}))

    return create_model(agent.name, **top_level_fields)


def generate_config_schema_for_agent(agent: "Agent") -> AnyDict:
    model = generate_config_type_for_agent(agent)
    schema = model.model_json_schema()
    schema = t.cast("AnyDict", jsonref.replace_refs(schema, proxies=False, lazy_load=False))
    schema.pop("$defs", None)  # Remove $defs if present
    return schema


def hydrate_agent(agent_blueprint: "Agent", config: BaseModel) -> "Agent":
    """
    Creates a new, fully configured Agent instance by applying the parsed
    CLI configuration to a blueprint agent.

    Args:
        agent_blueprint: The original, partially configured agent from the user's file.
        config: The Pydantic model instance containing all parsed CLI arguments.

    Returns:
        A new Agent instance, ready to be executed.
    """
    # Start with a deep copy
    hydrated_agent = deepcopy(agent_blueprint)
    config_dict = config.model_dump()

    # 1 - Agent core settings
    for field, value in config_dict.items():
        if hasattr(hydrated_agent, field):
            setattr(hydrated_agent, field, value)

    # 2 - Hydrate Tools and Hooks
    for group_name in ["tools", "hooks"]:
        new_component_list = []
        blueprint_list = getattr(agent_blueprint, group_name, [])
        group_config = config_dict.get(group_name, {})

        for component in blueprint_list:
            target_obj = None
            original_args: AnyDict = {}
            component_id = ""

            # Find the original factory/class for this component.
            if factory := getattr(component, CONFIGURABLE_FACTORY_ATTR, None):
                target_obj = factory
                original_args = getattr(component, CONFIGURABLE_ARGS_ATTR, {})
                component_id = target_obj.__name__
            elif hasattr(type(component), CONFIGURABLE_ATTR):  # It's a class instance
                target_obj = type(component)
                component_id = target_obj.__name__

            # If we found a configurable component and have CLI config for it...
            logger.debug(
                f"Processing {component_id} with config: {group_config.get(component_id, {})}"
            )
            if target_obj and component_id in group_config:
                # Merge original args with CLI overrides (CLI wins).
                merged_args = original_args.copy()
                merged_args.update(group_config[component_id])

                # Re-create the component by calling its factory/class with the new args.
                new_component = target_obj(**merged_args)
                new_component_list.append(new_component)
            else:
                # This component wasn't configured, so we keep the original.
                new_component_list.append(component)

        # Replace the agent's tool/hook list with the newly hydrated one.
        setattr(hydrated_agent, group_name, new_component_list)

    return hydrated_agent
