import contextlib
import inspect
import typing as t

import jsonref  # type: ignore[import-untyped]
import typing_extensions as te
import yaml
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, Field, create_model
from pydantic_core import PydanticUndefined

from dreadnode.common_types import AnyDict, JsonDict
from dreadnode.meta.config import Component, ConfigInfo, Model
from dreadnode.util import get_obj_name, safe_issubclass, warn_at_user_stacklevel


class IntrospectionWarning(UserWarning):
    """Warnings related to introspection and config model generation."""


def get_config_model(blueprint: t.Any, name: str = "config") -> type[PydanticBaseModel]:
    """
    Generates a Pydantic BaseModel type from a blueprint instance (Model or Component).

    This model type describes the configuration options for the blueprint. An instantiated
    instance of this model can be used in hydration to reconfigure the object tree on the fly.

    Args:
        blueprint: The blueprint instance (Model or Component) to generate the config model from.
        name: The name of the config model.

    Returns:
        The generated Pydantic BaseModel type or None if no configurable fields were found.
    """
    try:
        return _get_config_model(blueprint, name=name)
    except Exception as e:  # noqa: BLE001
        warn_at_user_stacklevel(
            f"Failed to generate config model for {blueprint!r}: {e}", IntrospectionWarning
        )
        return create_model(name)  # empty model


def _is_configurable(obj: t.Any) -> bool:
    """Check if the object is a Model or Component, or contains one inside a list/dict."""
    if isinstance(obj, Model | Component):
        return True

    if isinstance(obj, list | tuple) and any(isinstance(i, Model | Component) for i in obj):
        return True

    if isinstance(obj, dict) and any(isinstance(v, Model | Component) for v in obj.values()):  # noqa: SIM103
        return True

    return False


def _get_config_model(blueprint: t.Any, name: str = "config") -> type[PydanticBaseModel]:
    """
    Generates a Pydantic BaseModel type from a blueprint instance (Model or Component).

    This model type describes the configuration options for the blueprint. An instantiated
    instance of this model can be used in hydration to reconfigure the object tree on the fly.

    Args:
        blueprint: The blueprint instance (Model or Component) to generate the config model from.
        name: The name of the config model.

    Returns:
        The generated Pydantic BaseModel type or None if no configurable fields were found.
    """
    fields: AnyDict = {}

    if isinstance(blueprint, Model):
        model_config: dict[str, ConfigInfo] = getattr(blueprint, "__dn_config__", {})
        for field_name in blueprint.__class__.model_fields:
            obj = getattr(blueprint, field_name)
            annotation = blueprint.__annotations__.get(field_name, t.Any)
            param_info = model_config.get(field_name)

            if not _is_configurable(obj) and not param_info:
                continue

            field_type, default = _resolve_type_and_default(obj, annotation, name=field_name)
            if param_info and param_info.expose_as:
                field_type = param_info.expose_as

            if safe_issubclass(field_type, PydanticBaseModel) and not field_type.model_fields:
                continue

            field_kwargs = {
                "description": "",
                **(param_info.field_kwargs if param_info else {}),
                "default": default,
            }
            field_kwargs.pop("default_factory", None)
            fields[field_name] = (field_type, Field(**field_kwargs))

    elif isinstance(blueprint, Component):
        for param_name, param_info in blueprint.__dn_param_config__.items():
            if not isinstance(param_info, ConfigInfo):
                raise TypeError(
                    f"Expected ConfigInfo for parameter '{param_name}', got {type(param_info)}"
                )

            obj = param_info.field_kwargs.get("default")
            param_sig = blueprint.signature.parameters.get(param_name)
            annotation = (
                param_info.expose_as or param_sig.annotation
                if param_sig and param_sig.annotation is not inspect.Parameter.empty
                else t.Any
            )

            # If this param is defined with a default factory and no current
            # default, skip it as we don't want it's type polluting the model
            # (we wouldn't be able to hydrate it anyways)
            #
            # example: `def function(foo: Thing = Param(default_factory=Thing))`

            if (
                obj in (Ellipsis, PydanticUndefined)
                and "default_factory" in param_info.field_kwargs
            ):
                continue

            param_type, default = _resolve_type_and_default(obj, annotation, name=param_name)
            param_type = param_info.expose_as or param_type

            if safe_issubclass(param_type, PydanticBaseModel) and not param_type.model_fields:
                continue

            field_kwargs = {"description": "", **param_info.field_kwargs, "default": default}
            fields[param_name] = (param_type, Field(**field_kwargs))

        for attr_name, obj in _find_nested_configurables(blueprint).items():
            attr_type, default = _resolve_type_and_default(obj, t.Any, name=attr_name)

            if safe_issubclass(attr_type, PydanticBaseModel) and not attr_type.model_fields:
                continue

            fields[attr_name] = (attr_type, Field(default=default))

    return create_model(name, **fields, __config__=ConfigDict(arbitrary_types_allowed=True))


def get_model_schema(model: type[PydanticBaseModel]) -> AnyDict:
    schema = model.model_json_schema()
    schema = t.cast("AnyDict", jsonref.replace_refs(schema, proxies=False, lazy_load=False))
    schema.pop("$defs", None)  # Remove $defs if present
    return schema


def get_config_schema(blueprint: t.Any) -> AnyDict:
    config_model = get_config_model(blueprint)
    if config_model is None:
        return {}
    return get_model_schema(config_model)


def flatten_model(
    model: PydanticBaseModel, prefix: str = "", *, skip_none: bool = True
) -> dict[str, t.Any]:
    """
    Collapses a Pydantic model instance into a flat dictionary.

    This function recursively processes a Pydantic model instance. Nested
    Pydantic models have their keys concatenated with a dot ('.'), mirroring
    how libraries like cyclopts handle nested model arguments.

    The flattening stops when it encounters a value that is not an instance
    of a Pydantic BaseModel (e.g., a primitive type, list, or a plain dict).

    Args:
        model: The Pydantic BaseModel instance to flatten.
        prefix: An internal parameter used for building keys during recursion.
        skip_none: If True, fields with None values are omitted from the result.

    Returns:
        A flat dictionary representing the model's configuration.
    """
    flat_dict: dict[str, t.Any] = {}

    # Iterate through all fields defined in the model
    for field_name in model.__class__.model_fields:
        value = getattr(model, field_name)
        new_key = f"{prefix}.{field_name}" if prefix else field_name

        # It's a nested config model, so we recurse deeper
        if isinstance(value, PydanticBaseModel):
            nested_flat_dict = flatten_model(value, prefix=new_key)
            flat_dict.update(nested_flat_dict)
        else:
            flat_dict[new_key] = value

    if skip_none:
        flat_dict = {k: v for k, v in flat_dict.items() if v is not None}

    return flat_dict


def get_inputs_and_params_from_config_model(
    model: PydanticBaseModel,
    prefix: str = "",
    *,
    skip_none: bool = True,
    exclude: set[str] | None = None,
) -> tuple[AnyDict, JsonDict]:
    exclude = exclude or set()
    inputs: AnyDict = {}
    params: JsonDict = {}

    for field_name in model.__class__.model_fields:
        if field_name in exclude:
            continue

        value = getattr(model, field_name)
        field_name_or_alias = model.__class__.model_fields[field_name].alias or field_name
        new_key = f"{prefix}.{field_name_or_alias}" if prefix else field_name_or_alias

        # It's a nested config model, so we recurse deeper
        if isinstance(value, PydanticBaseModel):
            nested_inputs, nested_params = get_inputs_and_params_from_config_model(
                value, prefix=new_key
            )
            inputs.update(nested_inputs)
            params.update(nested_params)
        elif isinstance(value, int | float | str | bool | None):
            params[new_key] = value
        else:
            inputs[new_key] = value

    if skip_none:
        inputs = {k: v for k, v in inputs.items() if v is not None}
        params = {k: v for k, v in params.items() if v is not None}

    return inputs, params


def _find_nested_configurables(obj: t.Any) -> dict[str, t.Any]:
    if isinstance(obj, str | int | float | bool | type(None) | type) or not hasattr(
        obj, "__dict__"
    ):
        return {}

    discovered: dict[str, t.Any] = {}
    with contextlib.suppress(Exception):
        for attr_name, attr_value in obj.__dict__.items():
            if attr_name.startswith("__"):
                continue

            if _is_configurable(attr_value):
                discovered[attr_name] = attr_value

    return discovered


def _find_nested_configurable(obj: t.Any) -> t.Any | None:
    if isinstance(obj, Component | Model):
        return obj

    if nested_configurables := _find_nested_configurables(obj):
        return next(iter(nested_configurables.values()))

    return None


def _resolve_type_and_default(obj: t.Any, annotation: t.Any, name: str) -> tuple[type, t.Any]:  # noqa: PLR0912, PLR0915
    """
    Resolve an arbitrary object into it's type and default value.

    This includes handling nested structures like lists, tuples, and dictionaries.

    Args:
        obj: The object to resolve.
        annotation: The type annotation associated with the object.
        name: The name of the field (for error messages).

    Returns:
        A tuple containing the resolved type and default value.
    """
    obj_type: type = t.cast("type", annotation)
    obj_default = obj
    nested_fields: AnyDict = {}
    nested_default: t.Any
    contains_configurable = False

    if isinstance(obj, list | tuple):
        used_names = set()

        for item in obj:
            if not isinstance(item, Model | Component) and not (
                item := _find_nested_configurable(item)
            ):
                continue

            contains_configurable = True
            item_name = get_obj_name(item, short=True, clean=True)

            suffix = 1
            while item_name in used_names:
                item_name = f"{item_name}_{suffix}"
                suffix += 1
            used_names.add(item_name)

            nested_model = get_config_model(item, f"{name}_{item_name}")
            if not nested_model.model_fields:
                continue

            nested_default = Ellipsis
            with contextlib.suppress(Exception):
                nested_default = nested_model()
            nested_fields[item_name] = (
                nested_model,
                Field(default=nested_default, description=""),
            )

        if contains_configurable or nested_fields:
            obj_type = create_model(name, **nested_fields)
            obj_default = Ellipsis
            with contextlib.suppress(Exception):
                obj_default = obj_type()

    elif isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(value, Model | Component) and not (
                value := _find_nested_configurable(value)
            ):
                continue

            contains_configurable = True
            nested_model = get_config_model(value, f"{name}_{key}")
            if nested_model.model_fields:
                nested_default = Ellipsis
                with contextlib.suppress(Exception):
                    nested_default = nested_model()
                nested_fields[key] = (nested_model, Field(default=nested_default))

        if contains_configurable or nested_fields:
            obj_type = create_model(name, **nested_fields)
            obj_default = Ellipsis
            with contextlib.suppress(Exception):
                obj_default = obj_type()

    elif isinstance(obj, Model | Component):
        obj_type = get_config_model(obj, name)
        obj_default = Ellipsis
        with contextlib.suppress(Exception):
            obj_default = obj_type()

    # If our annotation is a TypeVar, override as Any
    # to prevent Pydantic from freaking out
    if isinstance(obj_type, t.TypeVar | te.TypeVar):
        obj_type = t.Any

    return obj_type, obj_default


# Config formatting


def generate_config_yaml(blueprint: t.Any, *, comments: bool = True) -> str:
    """
    Generates a configuration file example from a blueprint object
    by interpreting its JSON schema.

    Args:
        blueprint: The component, agent, eval, or task to generate a config for.
        comments: Whether to include comments in the generated YAML.

    Returns:
        A string containing the configuration example.
    """
    schema = get_model_schema(get_config_model(blueprint))

    properties = schema.get("properties")
    if not properties:
        return "# This object has no configurable parameters."

    return _generate_config_yaml(properties, comments=comments)


def _generate_config_yaml(properties: AnyDict, indent: int = 0, *, comments: bool = False) -> str:
    output = []
    indent_str = "  " * indent

    for name, prop_schema in properties.items():
        if comments:
            comment_parts = []
            if "description" in prop_schema:
                comment_parts.append(prop_schema["description"])
            if "enum" in prop_schema:
                comment_parts.append(f"({', '.join(map(str, prop_schema['enum']))})")

            comment = ", ".join(comment_parts) if comment_parts else ""
            if comment:
                output.append(f"{indent_str}# {comment}")

        default_value = prop_schema.get("default")

        if prop_schema.get("type") == "object" and "properties" in prop_schema:
            output.append(f"{indent_str}{name}:")
            output.append(
                _generate_config_yaml(
                    prop_schema["properties"], indent=indent + 1, comments=comments
                )
            )
        else:
            value_str = yaml.dump({name: default_value}, sort_keys=False).strip().split(": ", 1)[1]
            output.append(f"{indent_str}{name}: {value_str}")

        if indent == 0:
            output.append("")

    return "\n".join(output)
