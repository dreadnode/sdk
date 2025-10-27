import contextlib
import copy
import typing as t

from loguru import logger
from pydantic import BaseModel as PydanticBaseModel

from dreadnode.common_types import AnyDict
from dreadnode.meta.config import Component, ConfigInfo, Model
from dreadnode.util import get_obj_name, warn_at_user_stacklevel

T = t.TypeVar("T")


class HydrationWarning(UserWarning):
    """Warning related to object hydration."""


def hydrate(blueprint: T, config: PydanticBaseModel | AnyDict) -> T:
    """
    Hydrates a blueprint instance by applying static configuration values
    from a Pydantic config model instance.

    This is a recursive, non-mutating process that returns a new, fully
    hydrated blueprint.
    """
    try:
        config_data = config.model_dump() if isinstance(config, PydanticBaseModel) else config
        return t.cast("T", _hydrate_recursive(blueprint, config_data))
    except Exception as e:  # noqa: BLE001
        warn_at_user_stacklevel(
            f"Failed to hydrate {blueprint!r} with config {config!r}: {e}", HydrationWarning
        )
        logger.exception("Failed to hydrate object")
        return blueprint


def _hydrate_recursive(obj: t.Any, override: t.Any) -> t.Any:  # noqa: PLR0911, PLR0912
    if override is None:
        with contextlib.suppress(Exception):
            return copy.deepcopy(obj)
        return copy.copy(obj)

    override_is_dict = isinstance(override, dict)
    if isinstance(obj, Component) and override_is_dict:
        hydrated_component = obj.clone()
        hydrated_config = {}

        for name, config in obj.__dn_param_config__.items():
            original_default = config.field_kwargs.get("default")
            hydrated_default = _hydrate_recursive(original_default, override.pop(name, None))
            new_field_kwargs = config.field_kwargs.copy()
            new_field_kwargs["default"] = hydrated_default
            hydrated_config[name] = ConfigInfo(field_kwargs=new_field_kwargs)

        hydrated_component.__dn_param_config__ = hydrated_config

        # Assume anything left is an attribute override
        for name, override_value in override.items():
            with contextlib.suppress(Exception):
                if not hasattr(obj, name):
                    continue
                hydrated_attr = _hydrate_recursive(getattr(obj, name), override_value)
                setattr(hydrated_component, name, hydrated_attr)

        return hydrated_component

    if isinstance(obj, Model) and override_is_dict:
        # First, recursively hydrate nested objects in the current model
        current_data = {}
        for field_name, field_info in obj.__class__.model_fields.items():
            real_field_name = field_info.alias or field_name
            if hasattr(obj, field_name):
                current_val = getattr(obj, field_name)
                # Only hydrate if there's an override for this field
                if field_name in override:
                    hydrated_val = _hydrate_recursive(current_val, override[field_name])
                    current_data[real_field_name] = hydrated_val
                else:
                    current_data[real_field_name] = current_val

        # Add any override values that aren't currently in the model
        for key, override_val in override.items():
            if key not in current_data:
                current_data[key] = override_val

        # Use model_validate to create a new instance and trigger
        # any validators and model_post_init if defined

        try:
            return obj.__class__.model_validate(current_data)
        except Exception as e:  # noqa: BLE001
            warn_at_user_stacklevel(
                f"Validation failed during hydration of {obj.__class__.__name__}, hydration may not be complete: {e}. ",
                HydrationWarning,
            )
            updates = {k: v for k, v in current_data.items() if hasattr(obj, k)}
            return obj.model_copy(update=updates, deep=True)

    if isinstance(obj, list) and override_is_dict:
        hydrated_list = []
        for item in obj:
            # This assumes the overrides are a dict keyed by the component's name.
            # TODO(nick): Handle indexing extensions here (name_1, name_2, etc.)
            item_name = get_obj_name(item, short=True, clean=True)
            item_overrides = override.get(item_name)
            hydrated_list.append(_hydrate_recursive(item, item_overrides))
        return hydrated_list

    if isinstance(obj, dict) and override_is_dict:
        hydrated_dict = {}
        for key, item in obj.items():
            item_overrides = override.get(key)
            hydrated_dict[key] = _hydrate_recursive(item, item_overrides)
        return hydrated_dict

    if not isinstance(obj, str | int | float | bool | type(None) | type) and hasattr(
        obj, "__dict__"
    ):
        with contextlib.suppress(Exception):
            for attr_name, attr_value in obj.__dict__.items():
                if attr_name.startswith("__") or not isinstance(attr_value, Component | Model):
                    continue

                hydrated = _hydrate_recursive(attr_value, override)
                obj_copy = copy.copy(obj)
                setattr(obj_copy, attr_name, hydrated)
                return obj_copy

    return override
