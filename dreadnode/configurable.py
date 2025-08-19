import contextlib
import functools
import inspect
import types
import typing as t
from copy import copy, deepcopy
from dataclasses import dataclass

import jsonref  # type: ignore[import-untyped]
from loguru import logger
from pydantic import BaseModel, Field, TypeAdapter, create_model

from dreadnode.types import AnyDict

T = t.TypeVar("T")
TypeT = t.TypeVar("TypeT", bound=type)
CallableT = t.TypeVar("CallableT", bound=t.Callable[..., t.Any])
ItemT = t.TypeVar("ItemT", bound=t.Callable[..., t.Any] | type)

CONFIGURABLE_ATTR = "_configurable"
CONFIGURABLE_FIELDS_ATTR = "_configurable_fields"
CONFIGURABLE_FACTORY_ATTR = "_configurable_factory"
CONFIGURABLE_ARGS_ATTR = "_configurable_args"


def clone_config_attrs(old: t.Any, new: T) -> T:
    """
    Clones the configurable attributes from one object to another.
    """
    for attr in (
        CONFIGURABLE_ATTR,
        CONFIGURABLE_FIELDS_ATTR,
        CONFIGURABLE_FACTORY_ATTR,
        CONFIGURABLE_ARGS_ATTR,
    ):
        if hasattr(old, attr):
            setattr(new, attr, getattr(old, attr))
    return new


@t.overload
def configurable(obj: ItemT) -> ItemT: ...


@t.overload
def configurable(obj: list[str] | None = None) -> t.Callable[[ItemT], ItemT]: ...


def configurable(obj: ItemT | list[str] | None = None) -> ItemT | t.Callable[[ItemT], ItemT]:
    """
    A universal decorator to mark a class or a factory function as configurable.

    It can be used in several ways:

    1.  On a class to expose all of its friendly attributes:
        ```python
        @configurable
        class MyAgent(BaseModel):
            param1: str
        ```

    2.  On a class to expose a specific subset of attributes:
        ```python
        @configurable(["param1"])
        class MyAgent(BaseModel):
            param1: str
            param2: int # This will not be exposed
        ```

    3.  On a factory function to expose all its parameters:
        ```python
        @configurable
        def my_tool_factory(arg1: int) -> Tool:
            ...
        ```

    4.  On a factory function to expose a specific subset of parameters:
        ```python
        @configurable(["arg1"])
        def my_tool_factory(arg1: int, arg2: bool) -> Tool:
            ...
        ```

    5.  On a factory function to expose a subset of its parameters:
        ```python
        @configurable(["arg1"])
        def my_tool_factory(arg1: int, arg2: bool) -> Tool:
            ...
        ```

    5.  On a task to control which parameters are exposed:
        ```python
        @configurable(["model", "flag"])
        @task()
        def my_task(input: str, *, model: int, flag: bool):
            ...
    """

    exposed_fields: list[str] | bool = True
    if isinstance(obj, list):
        exposed_fields = obj

    def decorator(obj: ItemT) -> ItemT:
        # Tag the object with the primary configurable markers.
        setattr(obj, CONFIGURABLE_ATTR, True)
        setattr(obj, CONFIGURABLE_FIELDS_ATTR, exposed_fields)

        # If the decorated object is a class, our work is done. Just tag and return.
        if inspect.isclass(obj):
            return t.cast("ItemT", obj)

        if not callable(obj):
            raise TypeError(
                f"The @configurable decorator can only be applied to classes or functions, "
                f"not to objects of type {type(obj).__name__}."
            )

        if inspect.iscoroutinefunction(obj):

            async def factory_wrapper(*w_args: t.Any, **w_kwargs: t.Any) -> t.Any:
                result = await obj(*w_args, **w_kwargs)
                if callable(result) or hasattr(result, "__class__"):
                    try:
                        setattr(result, CONFIGURABLE_FACTORY_ATTR, obj)
                        bound_args = inspect.signature(obj).bind(*w_args, **w_kwargs)
                        bound_args.apply_defaults()
                        setattr(result, CONFIGURABLE_ARGS_ATTR, bound_args.arguments)
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"Error occurred while processing factory wrapper: {e}")

                return result

        else:

            def factory_wrapper(*w_args: t.Any, **w_kwargs: t.Any) -> t.Any:  # type: ignore[misc]
                result = obj(*w_args, **w_kwargs)
                if callable(result) or hasattr(result, "__class__"):
                    try:
                        setattr(result, CONFIGURABLE_FACTORY_ATTR, obj)
                        bound_args = inspect.signature(obj).bind(*w_args, **w_kwargs)
                        bound_args.apply_defaults()
                        setattr(result, CONFIGURABLE_ARGS_ATTR, bound_args.arguments)
                    except Exception as e:  # noqa: BLE001
                        logger.debug(f"Error occurred while processing factory wrapper: {e}")
                return result

        return t.cast("ItemT", functools.wraps(obj)(factory_wrapper))

    if callable(obj) and not isinstance(obj, list):
        return decorator(t.cast("ItemT", obj))

    return decorator


@dataclass
class ConfigurableSpec:
    obj: t.Any
    fields: list[str] | bool
    defaults: AnyDict
    components: dict[str, list[t.Any]]


def _is_configurable(obj: t.Any) -> bool:
    return (
        getattr(obj, CONFIGURABLE_ATTR, False)
        or getattr(type(obj), CONFIGURABLE_ATTR, False)
        or hasattr(obj, CONFIGURABLE_FACTORY_ATTR)
    )


PRIMITIVE_TYPES = {str, int, bool, float, type(None)}
PRIMITIVE_JSON_TYPES = {"string", "integer", "number", "boolean", "null"}


def _schema_is_primitive(schema: AnyDict) -> bool:  # noqa: PLR0911
    schema_type = schema.get("type")

    # Handle primitive types
    if schema_type in PRIMITIVE_JSON_TYPES:
        return True

    # Handle arrays
    if schema_type == "array":
        items = schema.get("items", {})
        return _schema_is_primitive(items)

    # Handle objects (dictionaries)
    if schema_type == "object":
        # Check if it's a simple key-value mapping (additionalProperties)
        additional_props = schema.get("additionalProperties")
        if additional_props is not None:
            if additional_props is True:
                return True  # Any additional properties allowed
            if isinstance(additional_props, dict):
                return _schema_is_primitive(additional_props)

        # Check if it has defined properties (this indicates a complex object)
        return "properties" in schema

    # Unions
    for union_key in ["anyOf", "oneOf", "allOf"]:
        if union_key in schema:
            union_schemas = schema[union_key]
            return all(_schema_is_primitive(s) for s in union_schemas)

    # Handle references ($ref) - these typically point to complex objects
    if "$ref" in schema:
        return False

    # Last check for enum, otherwise assume it's complex
    return "enum" in schema


def _is_primitive_type(annotation: t.Any) -> bool:
    """
    Checks if a type annotation is a primitive type we can handle in CLI/UI components.
    """

    # Robust path with full schema inspection

    with contextlib.suppress(Exception):
        adapter = TypeAdapter(annotation)
        return _schema_is_primitive(adapter.json_schema())

    # Fallback to manual type inspection

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


def _safe_issubclass(cls: t.Any, class_or_tuple: TypeT) -> t.TypeGuard[TypeT]:
    """Safely check if a class is a subclass of another class or tuple."""
    try:
        return isinstance(cls, type) and issubclass(cls, class_or_tuple)
    except TypeError:
        return False


def _get_name(obj: t.Any) -> str:
    """Safely retrieves the name of an object, falling back to its class name if necessary."""
    if hasattr(obj, "name"):
        return str(obj.name)
    if hasattr(obj, "__name__"):
        return str(obj.__name__)
    return str(type(obj).__name__)


def _make_config_model_fields(spec: ConfigurableSpec) -> dict[str, t.Any] | None:  # noqa: PLR0912
    # with contextlib.suppress(Exception):

    model_fields: AnyDict = {}
    if not spec.fields:
        return None

    # If the object is already a Pydantic model, use its fields directly
    if _safe_issubclass(spec.obj, BaseModel):
        instance = spec.defaults.get("__instance__")
        for field_name, field in spec.obj.model_fields.items():
            if isinstance(spec.fields, list) and field_name not in spec.fields:
                continue

            if not callable(instance) and hasattr(instance, field_name):
                field.default = getattr(instance, field_name)

            model_fields[field_name] = (field.annotation, field)

        return model_fields

    # If the object already has a __signature__, use that
    if hasattr(spec.obj, "__signature__"):
        signature = spec.obj.__signature__

    # Otherwise use inspect to get the signature
    else:

        @functools.wraps(spec.obj)
        def empty_func(*args, **kwargs):  # type: ignore [no-untyped-def] # noqa: ARG001
            return kwargs

        # If the object has annotations, use them directly
        if hasattr(spec.obj, "__annotations__"):
            empty_func.__annotations__ = spec.obj.__annotations__.copy()

        # Clear/filter the annotations to help reduce introspection errors
        empty_func.__annotations__.pop("__return__", None)
        if isinstance(spec.fields, list):
            empty_func.__annotations__ = {
                k: v for k, v in empty_func.__annotations__.items() if k in spec.fields
            }

        try:
            signature = inspect.signature(empty_func, eval_str=True)
        except (ValueError, TypeError, NameError):
            # print(f"Failed to inspect {obj.__name__}: {e}")
            return None

    for param in signature.parameters.values():
        if (
            isinstance(spec.fields, list) and param.name not in spec.fields
        ) or not _is_primitive_type(param.annotation):
            continue

        default_value = spec.defaults.get(param.name, param.default)
        instance = spec.defaults.get("__instance__")
        if not callable(instance) and hasattr(instance, param.name):
            default_value = getattr(instance, param.name)

        model_fields[param.name] = (
            param.annotation,
            Field(default=... if default_value is inspect.Parameter.empty else default_value),
        )

    return model_fields


def _resolve_configurable(obj: t.Any) -> ConfigurableSpec | None:
    target_obj = None
    defaults: AnyDict = {}

    if factory := getattr(obj, CONFIGURABLE_FACTORY_ATTR, None):
        target_obj = factory
        defaults = getattr(obj, CONFIGURABLE_ARGS_ATTR, {})
    elif getattr(obj, CONFIGURABLE_ATTR, False):
        target_obj = obj
    elif getattr(type(obj), CONFIGURABLE_ATTR, False):
        target_obj = type(obj)
        defaults = {"__instance__": obj}

    if target_obj is None:
        return None

    spec = ConfigurableSpec(
        obj=target_obj,
        fields=copy(getattr(target_obj, CONFIGURABLE_FIELDS_ATTR, True)),
        defaults=defaults,
        components={},
    )

    # For any configurable fields, check if they are also configurable
    # and move them to the components dict for special recursive handling.

    if isinstance(spec.fields, list):
        for field in spec.fields:
            field_obj = getattr(spec.obj, field, None)
            components = (
                field_obj if isinstance(field_obj, list) else [field_obj] if field_obj else []
            )
            configurable_components = [comp for comp in components if _is_configurable(comp)]
            if configurable_components:
                spec.components[field] = configurable_components
                spec.fields.remove(field)

    return spec


def make_config_type(obj: t.Any) -> type[BaseModel] | None:
    if (spec := _resolve_configurable(obj)) is None:
        return None

    top_level_fields = _make_config_model_fields(spec) or {}

    # For any nested configurable fields, recursively create models.

    for group_name, components in spec.components.items():
        nested_fields: AnyDict = {}
        for component in components:
            if model := make_config_type(component):
                component_name = getattr(component, "name", type(component).__name__)
                nested_fields[component_name] = (model, Field())

        if nested_fields:
            group_model = create_model(group_name, **nested_fields)
            top_level_fields[group_name] = (group_model, Field())

    return create_model(_get_name(spec.obj), **top_level_fields)


def get_model_schema(model: BaseModel) -> AnyDict:
    schema = model.model_json_schema()
    schema = t.cast("AnyDict", jsonref.replace_refs(schema, proxies=False, lazy_load=False))
    schema.pop("$defs", None)  # Remove $defs if present
    return schema


# Hydration


def _rebuild_configurable(component: t.Any, overrides: AnyDict) -> t.Any:
    # Case A: The component was created from a @configurable factory function.
    if factory := getattr(component, CONFIGURABLE_FACTORY_ATTR, None):
        original_args = getattr(component, CONFIGURABLE_ARGS_ATTR, {})
        # CLI overrides take precedence.
        merged_args = {**original_args, **overrides}
        # Re-call the factory with the merged arguments.
        return factory(**merged_args)

    # Case B: The component is an instance of a @configurable class.
    if getattr(type(component), CONFIGURABLE_ATTR, False):
        # We need to create a new instance of the class.
        # Start with the original object's attributes.
        original_args = {
            key: getattr(component, key)
            for key in component.model_fields
            if hasattr(component, key)
        }
        merged_args = {**original_args, **overrides}
        return type(component)(**merged_args)

    # This should not be reached if the component was found to be configurable.
    return component


def _hydrate_components(
    blueprint_list: list[t.Any], group_config: dict[str, AnyDict]
) -> list[t.Any]:
    """
    Hydrates a list of components (like tools or scorers) using config overrides.
    """
    new_component_list = []
    for component in blueprint_list:
        # First, we need to identify the component so we can find its config.
        # This relies on the same logic `generate_config_model` used.
        target_obj = None
        component_id = ""
        if factory := getattr(component, CONFIGURABLE_FACTORY_ATTR, None):
            target_obj = factory
            component_id = target_obj.__name__
        elif getattr(type(component), CONFIGURABLE_ATTR, False):
            target_obj = type(component)
            component_id = target_obj.__name__

        # If we found a configurable component AND the user provided config for it...
        if target_obj and component_id in group_config:
            # This component needs to be rebuilt.
            rebuilt_component = _rebuild_configurable(component, group_config[component_id])
            new_component_list.append(rebuilt_component)
        else:
            # This component was not configured via the CLI, so we keep the original.
            new_component_list.append(component)

    return new_component_list


def hydrate(
    blueprint: T,
    config: BaseModel,
    *,
    component_groups: dict[str, list[t.Any]] | None = None,
) -> T:
    """
    Creates a new, fully configured instance by applying CLI/config file
    settings to a blueprint object.

    This is the generic inverse of `generate_config_model`.

    Args:
        blueprint: The original, partially configured object from the user's file.
        config: The Pydantic model instance containing all parsed CLI arguments.
        component_groups: A dictionary mapping group names to the blueprint's
                          original list of components.

    Returns:
        A new, fully configured instance of the blueprint's type.
    """
    # Start with a deep copy of the blueprint to avoid modifying the original.
    hydrated_object = deepcopy(blueprint)
    config_dict = config.model_dump()

    # 1. Hydrate the root object's top-level fields.
    for field, value in config_dict.items():
        if hasattr(hydrated_object, field) and field not in (component_groups or {}):
            setattr(hydrated_object, field, value)

    component_groups = component_groups or {}

    # 2. Hydrate the component groups (e.g., "tools", "scorers").
    for group_name, blueprint_list in component_groups.items():
        if group_name not in config_dict:
            continue

        group_config = config_dict[group_name]
        new_component_list = _hydrate_components(blueprint_list, group_config)

        # Replace the component list on the hydrated object with the new one.
        setattr(hydrated_object, group_name, new_component_list)

    return hydrated_object
