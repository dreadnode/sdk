import contextlib
import typing as t
from copy import deepcopy

from pydantic import BaseModel as PydanticBaseModel

from dreadnode.meta.types import Component, ConfigInfo, Model
from dreadnode.types import AnyDict
from dreadnode.util import get_obj_name

T = t.TypeVar("T")


def hydrate(blueprint: T, config: PydanticBaseModel | AnyDict) -> T:
    """
    Hydrates a blueprint instance by applying static configuration values
    from a Pydantic config model instance.

    This is a recursive, non-mutating process that returns a new, fully
    hydrated blueprint.
    """
    config_data = (
        config.model_dump(exclude_unset=True) if isinstance(config, PydanticBaseModel) else config
    )
    return t.cast("T", _hydrate_recursive(blueprint, config_data))


def _hydrate_recursive(obj: t.Any, override: t.Any) -> t.Any:  # noqa: PLR0911
    if override is None:
        return deepcopy(obj)

    override_is_dict = isinstance(override, dict)
    if isinstance(obj, Component) and override_is_dict:
        hydrated_component = obj.clone()
        hydrated_config = {}

        for name, config in obj.__dn_param_config__.items():
            original_default = config.field_kwargs.get("default")
            hydrated_default = _hydrate_recursive(original_default, override.get(name))
            new_field_kwargs = config.field_kwargs.copy()
            new_field_kwargs["default"] = hydrated_default
            hydrated_config[name] = ConfigInfo(field_kwargs=new_field_kwargs)

        hydrated_component.__dn_param_config__ = hydrated_config

        for name, attr_info in obj.__dn_attr_config__.items():
            original_default = attr_info.field_kwargs.get("default")
            hydrated_default = _hydrate_recursive(original_default, override.get(name))
            setattr(hydrated_component, name, hydrated_default)

        return hydrated_component

    if isinstance(obj, Model) and override_is_dict:
        updates: AnyDict = {}
        for key, override_val in override.items():
            if hasattr(obj, key):
                current_val = getattr(obj, key)
                hydrated_attr = _hydrate_recursive(current_val, override_val)
                updates[key] = hydrated_attr

        with contextlib.suppress(Exception):
            return obj.model_copy(update=updates, deep=True)
        return obj.model_copy(update=updates)

    if isinstance(obj, list) and override_is_dict:
        hydrated_list = []
        for item in obj:
            # This assumes the overrides are a dict keyed by the component's name.
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

    return override
