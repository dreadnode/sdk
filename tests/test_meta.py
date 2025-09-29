import typing as t

import pytest
from pydantic import BaseModel, Field, PrivateAttr, ValidationError
from pydantic_core import PydanticUndefined

from dreadnode.meta.config import Component, Config, ConfigInfo, Model, component
from dreadnode.meta.hydrate import hydrate
from dreadnode.meta.introspect import get_config_model, get_config_schema

# ruff: noqa: N806

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


class Agent(Model):
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
    model_fields = Agent.model_fields

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
    assert hasattr(Agent, "__dn_config__")
    internal_params = Agent.__dn_config__

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
    agent = Agent(name="MyAgent")
    assert agent.retries == 3
    assert agent.name == "MyAgent"
    assert agent.optional_setting is None

    # Invalid case for `retries`
    with pytest.raises(ValidationError):
        Agent(name="MyAgent", retries=10)  # > 5

    # Invalid case for `name`
    with pytest.raises(ValidationError):
        Agent(name="")  # min_length=1

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
    version: int = Config(default=1)
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


def test_get_model_handles_primitive_list() -> None:
    class PrimitiveList(Model):
        items: list[str] = Config(default_factory=list)

    blueprint = PrimitiveList(items=["a", "b", "c"])
    ConfigModel = get_config_model(blueprint)

    fields = ConfigModel.model_fields
    assert "items" in fields

    assert fields["items"].annotation == list[str]
    assert fields["items"].default == ["a", "b", "c"]


def test_get_model_handles_primitive_dict() -> None:
    class PrimitiveDict(Model):
        mapping: dict[str, int] = Config(default_factory=dict)

    blueprint = PrimitiveDict(mapping={"one": 1, "two": 2})
    ConfigModel = get_config_model(blueprint)

    fields = ConfigModel.model_fields
    assert "mapping" in fields

    assert fields["mapping"].annotation == dict[str, int]
    assert fields["mapping"].default == {"one": 1, "two": 2}


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
            "items": {
                "properties": {
                    "component_with_default": {
                        "properties": {
                            "model": {
                                "default": "gpt-4",
                                "description": "",
                                "title": "Model",
                                "type": "string",
                            }
                        },
                        "title": "items_component_with_default",
                        "type": "object",
                    }
                },
                "title": "items",
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
            "mapping": {
                "properties": {
                    "component": {
                        "properties": {
                            "name": {"description": "", "title": "Name", "type": "string"}
                        },
                        "required": ["name"],
                        "title": "mapping_component",
                        "type": "object",
                    }
                },
                "required": ["component"],
                "title": "mapping",
                "type": "object",
            },
            "name": {"default": "override", "title": "Name", "type": "string"},
            "func": {
                "properties": {
                    "model": {
                        "default": "gpt-4o-mini",
                        "description": "",
                        "title": "Model",
                        "type": "string",
                    }
                },
                "title": "func",
                "type": "object",
            },
        },
        "required": ["mapping"],
        "title": "config",
        "type": "object",
    }

    assert get_config_schema(empty_blueprint) == {
        "properties": {
            "version": {"default": 1, "title": "Version", "type": "integer"},
            "sub": {
                "properties": {
                    "parameter": {"default": True, "title": "Parameter", "type": "boolean"}
                },
                "title": "sub",
                "type": "object",
            },
            "name": {"default": "empty", "title": "Name", "type": "string"},
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


#
# Annotations
#

# Test Components with Annotation-Based Config


@component
def component_annotation_only(
    name: t.Annotated[str, Config(help="Required name parameter")],
) -> str:
    return f"Hello {name}"


@component
def component_annotation_with_validation(
    count: t.Annotated[int, Config(help="Must be positive", gt=0, le=100)],
) -> int:
    return count * 2


@component
def component_annotation_with_regular_default(
    # Key test case: annotation Config + regular default value
    name: t.Annotated[str, Config(help="Name parameter")] = "default_name",
) -> str:
    return f"Hello {name}"


@component
def component_mixed_config(
    value: t.Annotated[float, Config(help="Base help", gt=0)] = Config(
        1.0, help="Override help", le=100
    ),
) -> float:
    return value


@component
def component_annotation_and_traditional(
    required: t.Annotated[str, Config(help="Required via annotation")],
    optional: int = Config(42, help="Optional via assignment"),
    # Another key case: annotation + regular default
    mixed: t.Annotated[str, Config(help="Mixed parameter")] = "regular_default",
) -> str:
    return f"{required}: {optional}: {mixed}"


# Test Models with Annotation-Based Config


class ModelAnnotationOnly(Model):
    name: t.Annotated[str, Config(help="Required name field")]


class ModelAnnotationWithDefault(Model):
    # This is the key case: annotation Config + regular default value
    name: t.Annotated[str, Config(help="Name with annotation")] = "default_name"
    count: t.Annotated[int, Config(help="Count with validation", gt=0)] = 42


class ModelMixedConfig(Model):
    # Pure annotation
    required_field: t.Annotated[str, Config(help="Required field")]

    # Traditional assignment
    optional_field: int = Config(42, help="Optional field")

    # Merged config - annotation + assignment (both are ConfigInfo)
    merged_field: t.Annotated[float, Config(help="Base help", gt=0)] = Config(
        1.0, help="Override help", le=100
    )

    # Annotation + regular default (the case we were missing)
    annotation_with_default: t.Annotated[str, Config(help="From annotation")] = "regular_default"


class ModelAnnotationWithValidation(Model):
    count: t.Annotated[int, Config(help="Positive integer", gt=0, le=100)]
    email: t.Annotated[str, Config(help="Valid email", pattern=r"^[^@]+@[^@]+\.[^@]+$")]


# Tests for Component Annotation Discovery


def test_component_annotation_only_discovery():
    """Test that pure annotation-based config is discovered correctly."""
    assert hasattr(component_annotation_only, "__dn_param_config__")
    params = component_annotation_only.__dn_param_config__

    assert "name" in params
    config_info = params["name"]
    assert isinstance(config_info, ConfigInfo)
    assert config_info.field_kwargs["description"] == "Required name parameter"
    # Should be required (no default value)
    assert config_info.field_kwargs.get("default", Ellipsis) is Ellipsis


def test_component_annotation_with_regular_default():
    """Test that annotation Config gets merged with regular default values."""
    params = component_annotation_with_regular_default.__dn_param_config__

    assert "name" in params
    config_info = params["name"]
    assert isinstance(config_info, ConfigInfo)

    # Should have description from annotation
    assert config_info.field_kwargs["description"] == "Name parameter"
    # Should have default value from parameter default
    assert config_info.field_kwargs["default"] == "default_name"


def test_component_mixed_patterns_with_defaults():
    """Test component with annotation+default alongside traditional patterns."""
    params = component_annotation_and_traditional.__dn_param_config__

    # Pure annotation (required)
    assert "required" in params
    required_config = params["required"]
    assert required_config.field_kwargs["description"] == "Required via annotation"
    assert required_config.field_kwargs.get("default", Ellipsis) is Ellipsis

    # Traditional Config assignment
    assert "optional" in params
    optional_config = params["optional"]
    assert optional_config.field_kwargs["description"] == "Optional via assignment"
    assert optional_config.field_kwargs["default"] == 42

    # Annotation + regular default (the key case)
    assert "mixed" in params
    mixed_config = params["mixed"]
    assert mixed_config.field_kwargs["description"] == "Mixed parameter"
    assert mixed_config.field_kwargs["default"] == "regular_default"


def test_component_mixed_config_discovery():
    """Test that annotation and assignment configs are merged correctly."""
    params = component_mixed_config.__dn_param_config__

    assert "value" in params
    config_info = params["value"]

    # Assignment should override annotation for conflicting fields
    assert config_info.field_kwargs["description"] == "Override help"
    assert config_info.field_kwargs["default"] == 1.0

    # Non-conflicting fields should be merged
    assert config_info.field_kwargs["gt"] == 0  # From annotation
    assert config_info.field_kwargs["le"] == 100  # From assignment


def test_component_mixed_patterns():
    """Test component with both annotation and traditional parameter patterns."""
    params = component_annotation_and_traditional.__dn_param_config__

    # Annotation-only parameter
    assert "required" in params
    required_config = params["required"]
    assert required_config.field_kwargs["description"] == "Required via annotation"
    assert required_config.field_kwargs.get("default", Ellipsis) is Ellipsis

    # Traditional assignment parameter
    assert "optional" in params
    optional_config = params["optional"]
    assert optional_config.field_kwargs["description"] == "Optional via assignment"
    assert optional_config.field_kwargs["default"] == 42


# Tests for Component Function Calls


def test_component_annotation_only_call():
    """Test that annotation-only components work correctly when called."""
    # Should work with explicit argument
    result = component_annotation_only("Alice")
    assert result == "Hello Alice"

    # Should fail without required argument
    with pytest.raises(TypeError, match="Missing required"):
        component_annotation_only()


def test_component_annotation_with_regular_default_call():
    """Test that annotation+default components use defaults correctly."""
    # Should use default value when no argument provided
    result = component_annotation_with_regular_default()
    assert result == "Hello default_name"

    # Should accept override
    result = component_annotation_with_regular_default("Alice")
    assert result == "Hello Alice"


def test_component_mixed_patterns_call():
    """Test calling component with all different parameter patterns."""
    # Should fail without required parameter
    with pytest.raises(TypeError, match="Missing required"):
        component_annotation_and_traditional()

    # Should work with just required parameter (others use defaults)
    result = component_annotation_and_traditional("test")
    assert result == "test: 42: regular_default"

    # Should work with all parameters overridden
    result = component_annotation_and_traditional("test", 99, "override")
    assert result == "test: 99: override"


# Tests for Model Annotation Discovery


def test_model_annotation_with_regular_default():
    """Test that Model handles annotation Config + regular default values."""
    config = ModelAnnotationWithDefault.__dn_config__

    # Check annotation + regular default case
    assert "name" in config
    name_info = config["name"]
    assert name_info.field_kwargs["description"] == "Name with annotation"
    assert name_info.field_kwargs["default"] == "default_name"

    # Check annotation with validation + regular default
    assert "count" in config
    count_info = config["count"]
    assert count_info.field_kwargs["description"] == "Count with validation"
    assert count_info.field_kwargs["gt"] == 0
    assert count_info.field_kwargs["default"] == 42


def test_model_annotation_with_regular_default_pydantic_fields():
    """Test that Pydantic Fields are created correctly for annotation+default."""
    model_fields = ModelAnnotationWithDefault.model_fields

    # Check that Pydantic Fields have correct defaults
    assert "name" in model_fields
    assert model_fields["name"].default == "default_name"
    assert model_fields["name"].description == "Name with annotation"

    assert "count" in model_fields
    assert model_fields["count"].default == 42
    assert model_fields["count"].metadata[0].gt == 0


def test_model_annotation_with_regular_default_instantiation():
    """Test that Model instances work correctly with annotation+default."""
    # Should use defaults
    instance = ModelAnnotationWithDefault()
    assert instance.name == "default_name"
    assert instance.count == 42

    # Should accept overrides
    instance2 = ModelAnnotationWithDefault(name="override", count=99)
    assert instance2.name == "override"
    assert instance2.count == 99

    # Should validate (count > 0)
    with pytest.raises(ValidationError):
        ModelAnnotationWithDefault(count=0)


def test_model_mixed_config_discovery():
    """Test that Model handles mixed annotation and assignment configs."""
    config = ModelMixedConfig.__dn_config__

    # Pure annotation (required)
    assert "required_field" in config
    required_info = config["required_field"]
    assert required_info.field_kwargs["description"] == "Required field"
    assert required_info.field_kwargs.get("default", Ellipsis) is Ellipsis

    # Traditional assignment
    assert "optional_field" in config
    optional_info = config["optional_field"]
    assert optional_info.field_kwargs["description"] == "Optional field"
    assert optional_info.field_kwargs["default"] == 42

    # Merged config (both are ConfigInfo)
    assert "merged_field" in config
    merged_info = config["merged_field"]
    assert merged_info.field_kwargs["description"] == "Override help"  # Assignment wins
    assert merged_info.field_kwargs["default"] == 1.0  # From assignment
    assert merged_info.field_kwargs["gt"] == 0  # From annotation
    assert merged_info.field_kwargs["le"] == 100  # From assignment

    # Annotation + regular default (key test case)
    assert "annotation_with_default" in config
    mixed_default_info = config["annotation_with_default"]
    assert mixed_default_info.field_kwargs["description"] == "From annotation"
    assert mixed_default_info.field_kwargs["default"] == "regular_default"


def test_model_annotation_validation():
    """Test that annotation-based validation works in Model instances."""
    # Valid instance
    instance = ModelAnnotationWithValidation(count=50, email="test@example.com")
    assert instance.count == 50
    assert instance.email == "test@example.com"

    # Invalid count (violates gt=0)
    with pytest.raises(ValidationError):
        ModelAnnotationWithValidation(count=0, email="test@example.com")

    # Invalid count (violates le=100)
    with pytest.raises(ValidationError):
        ModelAnnotationWithValidation(count=101, email="test@example.com")

    # Invalid email (violates pattern)
    with pytest.raises(ValidationError):
        ModelAnnotationWithValidation(count=50, email="invalid-email")


# Tests for Introspection with Annotations


def test_get_config_model_annotation_only():
    """Test that introspection works with annotation-only configs."""
    ConfigModel = get_config_model(component_annotation_only, "TestConfig")

    fields = ConfigModel.model_fields
    assert "name" in fields
    assert fields["name"].annotation is str
    assert fields["name"].is_required()
    assert fields["name"].description == "Required name parameter"


def test_get_config_model_mixed_patterns():
    """Test that introspection handles mixed annotation/assignment patterns."""
    ConfigModel = get_config_model(component_annotation_and_traditional, "MixedConfig")

    fields = ConfigModel.model_fields

    # Annotation-only field should be required
    assert "required" in fields
    assert fields["required"].is_required()
    assert fields["required"].description == "Required via annotation"

    # Traditional field should have default
    assert "optional" in fields
    assert fields["optional"].default == 42
    assert fields["optional"].description == "Optional via assignment"


def test_get_config_model_for_annotation_model():
    """Test introspection of Model classes with annotation-based configs."""
    instance = ModelMixedConfig(required_field="test")
    ConfigModel = get_config_model(instance, "ModelConfig")

    fields = ConfigModel.model_fields

    # Required annotation field
    assert "required_field" in fields
    assert not fields["required_field"].is_required()

    # Optional assignment field
    assert "optional_field" in fields
    assert fields["optional_field"].default == 42

    # Merged field
    assert "merged_field" in fields
    assert fields["merged_field"].default == 1.0


# Tests for Hydration with Annotations


def test_hydrate_annotation_based_component():
    """Test that hydration works with annotation-based component configs."""
    # Create a component instance that can be configured
    component_instance = component_mixed_config

    ConfigModel = get_config_model(component_instance, "HydrateConfig")
    config = ConfigModel(value=5.0)

    hydrated = hydrate(component_instance, config)

    # Should be a new instance
    assert hydrated is not component_instance

    # Should have updated config
    params = hydrated.__dn_param_config__
    assert params["value"].field_kwargs["default"] == 5.0


def test_hydrate_annotation_based_model():
    """Test that hydration works with annotation-based model configs."""
    instance = ModelMixedConfig(required_field="original")

    ConfigModel = get_config_model(instance, "HydrateConfig")
    config = ConfigModel(required_field="hydrated", optional_field=99)

    hydrated = hydrate(instance, config)

    # Should be a new instance
    assert hydrated is not instance

    # Should have updated values
    assert hydrated.required_field == "hydrated"
    assert hydrated.optional_field == 99


# Edge Cases and Error Handling


def test_annotation_without_config_ignored():
    """Test that regular annotations without Config are ignored."""

    @component
    def regular_annotations(name: str, count: int = 42) -> str:
        return f"{name}: {count}"

    # Should only find the Config from the default, not from regular annotations
    params = regular_annotations.__dn_param_config__
    assert "name" not in params  # Regular annotation, no Config
    assert "count" not in params  # Regular default value, no Config


def test_expose_as_works_with_annotations():
    """Test that expose_as parameter works in annotation-based configs."""

    @component
    def with_expose_as(
        value: t.Annotated[str, Config(expose_as=int, help="Exposed as int")],
    ) -> str:
        return value

    ConfigModel = get_config_model(with_expose_as, "ExposeAsConfig")
    fields = ConfigModel.model_fields

    # Should be exposed as int, not str
    assert fields["value"].annotation is int


def test_multiple_config_metadata_in_annotation():
    """Test handling of multiple Config instances in annotation metadata (should use first)."""

    @component
    def multiple_configs(
        # This is an edge case - multiple Config instances in metadata
        value: t.Annotated[str, Config(help="First"), Config(help="Second")],
    ) -> str:
        return value

    params = multiple_configs.__dn_param_config__
    config_info = params["value"]

    # Should use the first Config found
    assert config_info.field_kwargs["description"] == "First"
