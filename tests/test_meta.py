import typing as t

import pytest
from pydantic import BaseModel, Field, PrivateAttr, ValidationError
from pydantic_core import PydanticUndefined

from dreadnode.meta.hydrate import hydrate
from dreadnode.meta.introspect import get_config_model, get_config_schema
from dreadnode.meta.types import Component, Config, ConfigInfo, Model, component

# ruff: noqa: PLR2004, N806

#
# Primitives
#


def test_param_creates_param_info_object() -> None:
    """Verify that Param() returns the internal ParamInfo container."""
    p = Config(default=10)
    assert isinstance(p, ConfigInfo)
    assert p.field_kwargs["default"] == 10


def test_param_handles_required_fields() -> None:
    """Verify that Param() with no default is captured correctly."""
    p = Config()
    # Pydantic's internal sentinel for required is Ellipsis (...)
    assert p.field_kwargs["default"] is ...


def test_param_handles_none_as_default() -> None:
    """Verify the critical bugfix: Param(default=None) is preserved."""
    p = Config(default=None)
    assert "default" in p.field_kwargs
    assert p.field_kwargs["default"] is None


def test_param_collects_pydantic_kwargs() -> None:
    """Verify that validation and metadata kwargs are collected."""
    p = Config(default=5, gt=0, le=10, description="A number")
    assert p.field_kwargs["gt"] == 0
    assert p.field_kwargs["le"] == 10
    assert p.field_kwargs["description"] == "A number"


def test_param_help_overrides_description() -> None:
    """Verify `help` is a convenient alias for `description`."""
    p = Config(help="Help text", description="This should be ignored")
    assert p.field_kwargs["description"] == "Help text"


def test_param_removes_own_kwargs() -> None:
    """Verify that `key` and `help` are not passed into field_kwargs."""
    p = Config(key="my_key", help="my_help")
    assert "key" not in p.field_kwargs
    assert "help" not in p.field_kwargs


class TestAgent(Model):
    # Public, configurable, with validation and a new default
    retries: int = Config(default=3, gt=0, le=5)

    # Public, configurable, required field
    name: str = Config(..., min_length=1)

    # Public, configurable, with an optional default of None
    optional_setting: str | None = Config(default=None)

    # Private, internal field that should be IGNORED by our system
    session_id: str = Field(default="abc-123")


def test_model_transforms_params_to_fields() -> None:
    """Verify that __init_subclass__ correctly creates Pydantic Fields."""
    # This is an introspection test, we look at the generated Pydantic model fields
    model_fields = TestAgent.model_fields

    assert "retries" in model_fields
    assert model_fields["retries"].default == 3
    assert model_fields["retries"].metadata[0].gt == 0
    assert model_fields["retries"].metadata[1].le == 5

    assert "name" in model_fields
    assert model_fields["name"].is_required()

    assert "optional_setting" in model_fields
    assert model_fields["optional_setting"].default is None


def test_model_stores_param_info_internally() -> None:
    """Verify that the original ParamInfo is stored for our introspection engine."""
    assert hasattr(TestAgent, "__dn_config__")
    internal_params = TestAgent.__dn_config__

    assert "retries" in internal_params
    assert isinstance(internal_params["retries"], ConfigInfo)
    assert internal_params["retries"].field_kwargs["default"] == 3

    assert "name" in internal_params
    # Private field should not be in our map
    assert "session_id" not in internal_params


# Excluded for now as I'm not sure whether we should keep it
# def test_model_includes_json_schema_attribute() -> None:
#     """Verify that the model includes the JSON schema attribute."""
#     json_schema_extra = TestAgent.__dn_config__["name"].field_kwargs["json_schema_extra"]
#     assert "__dn_param__" in json_schema_extra
#     assert json_schema_extra["__dn_param__"] is True


def test_model_validation_works_as_expected() -> None:
    """Verify that the final class is a fully functional Pydantic model."""
    # Valid case
    agent = TestAgent(name="MyAgent")
    assert agent.retries == 3
    assert agent.name == "MyAgent"
    assert agent.optional_setting is None

    # Invalid case for `retries`
    with pytest.raises(ValidationError):
        TestAgent(name="MyAgent", retries=10)  # > 5

    # Invalid case for `name`
    with pytest.raises(ValidationError):
        TestAgent(name="")  # min_length=1

    # Check that private field works as a normal Pydantic field
    assert agent.session_id == "abc-123"


@component
def task_required_args(prefix: str, suffix: str = Config()) -> str:
    return f"{prefix} {suffix}"


@component
def task_optional_args(
    # Public, configurable parameter
    model: str = Config("gpt-4", help="The model to use"),
    # Private parameter with a normal default
    temperature: float = 0.7,
) -> None:
    """A sample task function."""
    return f"Using {model} at temp {temperature}"


def test_component_decorator_wraps_function() -> None:
    """Verify that @component returns a Component instance."""
    assert isinstance(task_optional_args, Component)
    assert (
        task_optional_args.func.__name__ == "task_optional_args"
    )  # Check that it's wrapped correctly


def test_component_discovers_params() -> None:
    """Verify the Component wrapper finds Param objects in the signature."""
    assert hasattr(task_optional_args, "__dn_param_config__")
    params = task_optional_args.__dn_param_config__

    assert "model" in params
    assert "temperature" not in params  # Should be ignored

    model_param_info = params["model"]
    assert isinstance(model_param_info, ConfigInfo)
    assert model_param_info.field_kwargs["default"] == "gpt-4"

    assert hasattr(task_required_args, "__dn_param_config__")
    params = task_required_args.__dn_param_config__
    assert "prefix" not in params  # Should be ignored
    assert "suffix" in params

    suffix_param_info = params["suffix"]
    assert isinstance(suffix_param_info, ConfigInfo)
    assert suffix_param_info.field_kwargs["default"] == Ellipsis  # Required field


def test_component_with_params_creates_new_blueprint() -> None:
    """Verify that with_params creates a new, altered Component instance."""
    new_task_blueprint = task_optional_args.configure(model="gpt-4o-mini")

    # 1. Verify it's a new object, not a mutation
    assert new_task_blueprint is not task_optional_args
    assert new_task_blueprint.func is task_optional_args.func

    # 2. Verify the old blueprint is unchanged
    assert task_optional_args.__dn_param_config__["model"].field_kwargs["default"] == "gpt-4"

    # 3. Verify the new blueprint has the updated default
    new_params = new_task_blueprint.__dn_param_config__
    assert new_params["model"].field_kwargs["default"] == "gpt-4o-mini"


def test_component_remains_callable() -> None:
    """Verify the Component wrapper can still be called like a function."""
    result = task_optional_args()  # The injector would normally provide `model`
    assert result == "Using gpt-4 at temp 0.7"

    # Verify a modified blueprint is also callable
    new_task_blueprint = task_optional_args.configure(model="gpt-4o-mini")

    result_new = new_task_blueprint()
    assert result_new == "Using gpt-4o-mini at temp 0.7"


#
# Introspection
#


class Sub(Model):
    parameter: bool = Config(default=True)
    field: str = Field("foo")
    _private: float = PrivateAttr(default=1.0)


def not_a_component(foo: int) -> None:
    pass


@component
def component_without_config(required: int) -> None:
    pass


@component
def component_with_default(model: str = Config("gpt-4")) -> None:
    pass


@component
def component_with_required(name: str = Config()) -> None:
    pass


@component
def component_complex(
    positional: str,
    *,
    temperature: float = Config(0.7),
    sub: Sub = Config(default_factory=Sub),  # noqa: B008
    items: list[str] = Config(default_factory=list),  # noqa: B008
    name: str = Config("component_complex"),
) -> None:
    pass


class Thing(Model):
    name: str = Config()
    func: t.Callable[..., t.Any] = Config(default=not_a_component)
    items: list[t.Any] = Config(default_factory=list)
    mapping: dict[str, t.Any] = Config(default_factory=dict)
    version: int = Config(1)
    sub: Sub = Config(default_factory=Sub)
    other: bool = False


@pytest.fixture
def blueprint() -> Thing:
    return Thing(
        name="override",
        func=component_with_default.configure(model="gpt-4o-mini"),
        items=["item1", component_with_default],
        mapping={"key1": not_a_component, "component": component_with_required, "key3": 123},
        sub=Sub(parameter=False, field="bar"),
        other=True,
    )


@pytest.fixture
def empty_blueprint() -> Thing:
    return Thing(
        name="empty",
        func=component_without_config,
        items=["item1", not_a_component, component_without_config],
        mapping={"key1": not_a_component, "component": component_without_config},
        other=False,
    )


def test_get_model_for_component_with_default() -> None:
    """Verify schema generation for a standalone function component."""
    ConfigModel = get_config_model(component_with_default, "config")

    assert issubclass(ConfigModel, BaseModel)
    assert ConfigModel.__name__ == "config"

    fields = ConfigModel.model_fields
    assert "model" in fields
    assert fields["model"].annotation is str
    assert fields["model"].default == "gpt-4"

    assert ConfigModel().model == "gpt-4"

    updated = component_with_default.configure(model="gpt-4o-mini")
    UpdatedModel = get_config_model(updated, "updated")

    assert issubclass(UpdatedModel, BaseModel)
    assert UpdatedModel.__name__ == "updated"

    fields = UpdatedModel.model_fields
    assert "model" in fields
    assert fields["model"].annotation is str
    assert fields["model"].default == "gpt-4o-mini"

    assert UpdatedModel().model == "gpt-4o-mini"


def test_get_model_for_component_with_required() -> None:
    """Verify that a component taking another component as a param is handled."""
    ConfigModel = get_config_model(component_with_required, "task_config")

    fields = ConfigModel.model_fields
    assert "name" in fields
    assert fields["name"].annotation is str
    assert fields["name"].default is PydanticUndefined

    ConfigModel(name="test")

    with pytest.raises(ValidationError):
        ConfigModel()


def test_get_model_for_component_complex() -> None:
    """Verify that a complex component with multiple parameters is handled."""

    ConfigModel = get_config_model(component_complex, "task_config")

    fields = ConfigModel.model_fields

    assert "positional" not in fields

    assert "temperature" in fields
    assert fields["temperature"].default == 0.7

    assert "config" not in fields

    ConfigModel()
    assert ConfigModel(temperature=0.5).temperature == 0.5


def test_get_model_for_class_based_model() -> None:
    """Verify generation for a simple declarai.Model."""
    ConfigModel = get_config_model(Sub(), "class_config")

    assert issubclass(ConfigModel, BaseModel)
    assert ConfigModel.__name__ == "class_config"

    fields = ConfigModel.model_fields
    assert "parameter" in fields
    assert fields["parameter"].default is True
    assert "field" not in fields
    assert "_private" not in fields

    assert ConfigModel(parameter=False).parameter is False


def test_get_model_is_instance_aware(blueprint: Thing) -> None:
    """Verify instance values correctly override defaults."""
    ConfigModel = get_config_model(blueprint, "thing_config")

    assert issubclass(ConfigModel, BaseModel)
    assert ConfigModel.__name__ == "thing_config"

    fields = ConfigModel.model_fields

    assert fields["name"].default == "override"
    assert fields["version"].default == 1

    ComponentModel = fields["func"].annotation
    component_fields = ComponentModel.model_fields
    assert component_fields["model"].default == "gpt-4o-mini"

    assert "sub" in fields
    SubConfigModel = fields["sub"].annotation
    assert issubclass(SubConfigModel, BaseModel)
    assert SubConfigModel.__name__ == "sub"

    sub_config_fields = SubConfigModel.model_fields
    assert "parameter" in sub_config_fields
    assert sub_config_fields["parameter"].default is False
    assert "field" not in sub_config_fields
    assert "_private" not in sub_config_fields


def test_get_model_handles_heterogeneous_list(blueprint: Thing) -> None:
    """Verify that a list of different components is handled correctly."""
    ConfigModel = get_config_model(blueprint)

    fields = ConfigModel.model_fields
    assert "items" in fields

    ItemsModel = fields["items"].annotation
    assert issubclass(ItemsModel, BaseModel)
    assert ItemsModel.__name__ == "items"

    group_fields = ItemsModel.model_fields
    assert len(blueprint.items) == 2  # Two items in the list
    assert len(group_fields) == 1  # only one component
    assert "component_with_default" in group_fields

    ComponentModel = group_fields["component_with_default"].annotation
    assert issubclass(ComponentModel, BaseModel)
    assert ComponentModel.__name__ == "items_component_with_default"

    component_fields = ComponentModel.model_fields
    assert "model" in component_fields
    assert component_fields["model"].default == "gpt-4"


def test_get_model_handles_dictionary_group(blueprint: Thing) -> None:
    """Verify that a dictionary of components creates a nested model with correct keys."""
    ConfigModel = get_config_model(blueprint, "AgentConfig")

    fields = ConfigModel.model_fields
    assert "mapping" in fields

    MappingModel = fields["mapping"].annotation
    assert issubclass(MappingModel, BaseModel)
    assert MappingModel.__name__ == "mapping"

    group_fields = MappingModel.model_fields
    assert len(blueprint.mapping) == 3  # Three items in the dict
    assert len(group_fields) == 1  # Only one component
    assert "component" in group_fields

    ComponentModel = group_fields["component"].annotation
    assert issubclass(ComponentModel, BaseModel)
    assert ComponentModel.__name__ == "mapping_component"

    component_fields = ComponentModel.model_fields
    assert "name" in component_fields
    assert component_fields["name"].default is PydanticUndefined


def test_get_model_handles_non_configurable_component() -> None:
    """Verify that non-configurable components are handled correctly."""
    ConfigModel = get_config_model(component_without_config)
    assert not ConfigModel.model_fields


def test_get_config_schema(blueprint: Thing, empty_blueprint: Thing) -> None:
    """Verify full schema creation for blueprints"""
    assert get_config_schema(blueprint) == {
        "properties": {
            "name": {"default": "override", "title": "Name", "type": "string"},
            "func": {
                "properties": {
                    "model": {"default": "gpt-4o-mini", "title": "Model", "type": "string"}
                },
                "title": "func",
                "type": "object",
            },
            "items": {
                "properties": {
                    "component_with_default": {
                        "properties": {
                            "model": {"default": "gpt-4", "title": "Model", "type": "string"}
                        },
                        "title": "items_component_with_default",
                        "type": "object",
                    }
                },
                "title": "items",
                "type": "object",
            },
            "mapping": {
                "properties": {
                    "component": {
                        "properties": {"name": {"title": "Name", "type": "string"}},
                        "required": ["name"],
                        "title": "mapping_component",
                        "type": "object",
                    }
                },
                "required": ["component"],
                "title": "mapping",
                "type": "object",
            },
            "version": {"default": 1, "title": "Version", "type": "integer"},
            "sub": {
                "properties": {
                    "parameter": {"default": False, "title": "Parameter", "type": "boolean"}
                },
                "title": "sub",
                "type": "object",
            },
        },
        "required": ["mapping"],
        "title": "config",
        "type": "object",
    }

    assert get_config_schema(empty_blueprint) == {
        "properties": {
            "name": {"default": "empty", "title": "Name", "type": "string"},
            "version": {"default": 1, "title": "Version", "type": "integer"},
            "sub": {
                "properties": {
                    "parameter": {"default": True, "title": "Parameter", "type": "boolean"}
                },
                "title": "sub",
                "type": "object",
            },
        },
        "title": "config",
        "type": "object",
    }


def test_generated_model_can_be_instantiated(blueprint: Thing) -> None:
    """Ensure the generated model can be instantiated with its own defaults."""
    ConfigModel = get_config_model(blueprint, "AgentConfig")

    config = ConfigModel(mapping={"component": {"name": "test"}})
    assert config.name == "override"
    assert config.func.model == "gpt-4o-mini"
    assert config.items.component_with_default.model == "gpt-4"
    assert config.mapping.component.name == "test"
    assert config.sub.parameter is False
    assert config.version == 1

    with pytest.raises(ValidationError):
        ConfigModel()


#
# Hydration
#


def test_hydrate_returns_new_instance(blueprint: Thing) -> None:
    """Verify that hydrate performs a deep copy and does not mutate the original."""
    ConfigModel = get_config_model(blueprint)
    config_instance = ConfigModel(
        name="new name", mapping={"component": {"name": "override"}}
    )  # A simple override

    hydrated = hydrate(blueprint, config_instance)

    assert hydrated is not blueprint, "Hydrate should return a new instance"
    assert hydrated.sub is not blueprint.sub, "Nested models should also be new instances"
    assert blueprint.name == "override", "Original blueprint should be unchanged"


def test_hydrate_top_level_fields(blueprint: Thing) -> None:
    """Tests overriding simple, top-level parameters on the blueprint."""
    ConfigModel = get_config_model(blueprint)
    config_instance = ConfigModel(
        name="hydrated name", version=99, mapping={"component": {"name": "override"}}
    )

    hydrated = hydrate(blueprint, config_instance)

    assert hydrated.name == "hydrated name"
    assert hydrated.version == 99
    # Verify non-overridden value from the blueprint instance is preserved
    assert hydrated.sub.parameter is False


def test_hydrate_nested_model(blueprint: Thing) -> None:
    """Tests overriding fields on a nested Model."""
    ConfigModel = get_config_model(blueprint)
    config_instance = ConfigModel(
        sub={"parameter": True}, mapping={"component": {"name": "override"}}
    )

    hydrated = hydrate(blueprint, config_instance)

    assert blueprint.sub.parameter is False  # Original is untouched
    assert hydrated.sub.parameter is True


def test_hydrate_nested_component_parameter(blueprint: Thing) -> None:
    """Tests re-configuring a nested Component with new defaults."""
    ConfigModel = get_config_model(blueprint)
    config_instance = ConfigModel(
        func={"model": "hydrated-model"}, mapping={"component": {"name": "override"}}
    )

    hydrated = hydrate(blueprint, config_instance)

    # Verify original blueprint's component is untouched
    assert blueprint.func.__dn_param_config__["model"].field_kwargs["default"] == "gpt-4o-mini"

    # Verify the hydrated blueprint has a new, re-configured component
    hydrated_task = hydrated.func
    assert isinstance(hydrated_task, Component)
    assert hydrated_task is not blueprint.func  # It must be a new Component instance
    assert hydrated_task.func is blueprint.func.func  # But it wraps the same raw function
    assert hydrated_task.__dn_param_config__["model"].field_kwargs["default"] == "hydrated-model"


def test_hydrate_heterogeneous_list(blueprint: Thing) -> None:
    """Tests hydration of components within a list, preserving other elements."""
    ConfigModel = get_config_model(blueprint)
    # The key 'component-with-default' is derived from the component's name
    config_instance = ConfigModel(
        items={"component_with_default": {"model": "hydrated-in-list"}},
        mapping={"component": {"name": "override"}},
    )

    hydrated = hydrate(blueprint, config_instance)

    # Verify primitives and structure are preserved
    assert len(hydrated.items) == 2
    assert hydrated.items[0] == "item1"

    # Verify the component in the list was hydrated
    hydrated_component = hydrated.items[1]
    assert isinstance(hydrated_component, Component)
    assert (
        hydrated_component.__dn_param_config__["model"].field_kwargs["default"]
        == "hydrated-in-list"
    )

    # Verify the original list component is untouched
    original_component = blueprint.items[1]
    assert original_component.__dn_param_config__["model"].field_kwargs["default"] == "gpt-4"


def test_hydrate_heterogeneous_dict(blueprint: Thing) -> None:
    """Tests hydration of components within a dict, preserving other key-value pairs."""
    ConfigModel = get_config_model(blueprint)
    # The key 'component' matches the key in the blueprint's dictionary
    config_instance = ConfigModel(mapping={"component": {"name": "hydrated-required-name"}})

    hydrated = hydrate(blueprint, config_instance)

    # Verify primitives and structure are preserved
    assert len(hydrated.mapping) == 3
    assert hydrated.mapping["key1"] == not_a_component
    assert hydrated.mapping["key3"] == 123

    # Verify the component in the dict was hydrated
    hydrated_component = hydrated.mapping["component"]
    assert isinstance(hydrated_component, Component)
    # This was a required parameter, so its default was originally ...
    assert (
        hydrated_component.__dn_param_config__["name"].field_kwargs["default"]
        == "hydrated-required-name"
    )


def test_full_hydration_integration(blueprint: Thing) -> None:
    """
    An integration test that applies multiple, deeply nested overrides at once.
    """
    ConfigModel = get_config_model(blueprint)

    # A complex set of overrides, as if parsed from a rich config file or CLI
    config_instance = ConfigModel(
        name="Fully Hydrated Thing",
        version=42,
        sub={"parameter": True},
        func={"model": "claude-3-opus"},
        items={"component_with_default": {"model": "llama3-70b"}},
        mapping={"component": {"name": "final-required-name"}},
    )

    hydrated = hydrate(blueprint, config_instance)

    # --- Assert all hydrated values are correct ---

    # Top level
    assert hydrated.name == "Fully Hydrated Thing"
    assert hydrated.version == 42

    # Nested Model
    assert hydrated.sub.parameter is True

    # Nested Component
    assert hydrated.func.__dn_param_config__["model"].field_kwargs["default"] == "claude-3-opus"

    # Component in List
    hydrated_list_comp = hydrated.items[1]
    assert hydrated_list_comp.__dn_param_config__["model"].field_kwargs["default"] == "llama3-70b"
    assert hydrated.items[0] == "item1"  # Primitive preserved

    # Component in Dict
    hydrated_dict_comp = hydrated.mapping["component"]
    assert (
        hydrated_dict_comp.__dn_param_config__["name"].field_kwargs["default"]
        == "final-required-name"
    )
    assert hydrated.mapping["key1"] == not_a_component
    assert hydrated.mapping["key3"] == 123

    # --- Assert original blueprint is still pristine ---
    assert blueprint.name == "override"
    assert blueprint.version == 1
    assert blueprint.sub.parameter is False
    assert blueprint.func.__dn_param_config__["model"].field_kwargs["default"] == "gpt-4o-mini"
    assert blueprint.items[1].__dn_param_config__["model"].field_kwargs["default"] == "gpt-4"
    assert blueprint.mapping["component"].__dn_param_config__["name"].field_kwargs["default"] is ...
