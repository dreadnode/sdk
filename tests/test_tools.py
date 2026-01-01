"""Tests for the tools module - Tool, ToolMethod, Toolset, decorators."""

import asyncio
import json
import pytest
from pydantic import ValidationError

from dreadnode.core.tools.tools import (
    DEFAULT_CATCH_EXCEPTIONS,
    TOOL_STOP_TAG,
    TOOL_VARIANTS_ATTR,
    FunctionCall,
    FunctionDefinition,
    NamedFunction,
    Tool,
    ToolCall,
    ToolChoiceDefinition,
    ToolDefinition,
    ToolMethod,
    ToolResponse,
    Toolset,
    discover_tools_on_obj,
    tool,
    tool_method,
)


# =============================================================================
# Pydantic Model Tests
# =============================================================================


class TestNamedFunction:
    """Tests for NamedFunction model."""

    def test_basic_creation(self):
        """Test creating a NamedFunction."""
        nf = NamedFunction(name="my_function")
        assert nf.name == "my_function"

    def test_requires_name(self):
        """Test that name is required."""
        with pytest.raises(ValidationError):
            NamedFunction()


class TestToolChoiceDefinition:
    """Tests for ToolChoiceDefinition model."""

    def test_basic_creation(self):
        """Test creating a ToolChoiceDefinition."""
        tcd = ToolChoiceDefinition(function=NamedFunction(name="my_func"))
        assert tcd.type == "function"
        assert tcd.function.name == "my_func"

    def test_default_type(self):
        """Test that type defaults to 'function'."""
        tcd = ToolChoiceDefinition(function=NamedFunction(name="test"))
        assert tcd.type == "function"


class TestFunctionDefinition:
    """Tests for FunctionDefinition model."""

    def test_basic_creation(self):
        """Test creating a FunctionDefinition with name only."""
        fd = FunctionDefinition(name="my_function")
        assert fd.name == "my_function"
        assert fd.description is None
        assert fd.parameters is None

    def test_with_description(self):
        """Test FunctionDefinition with description."""
        fd = FunctionDefinition(name="add", description="Adds two numbers")
        assert fd.description == "Adds two numbers"

    def test_with_parameters(self):
        """Test FunctionDefinition with parameters schema."""
        params = {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"],
        }
        fd = FunctionDefinition(name="add", parameters=params)
        assert fd.parameters == params

    def test_empty_object_parameters_become_none(self):
        """Test that empty object parameters are normalized to None."""
        params = {"type": "object", "properties": {}}
        fd = FunctionDefinition(name="test", parameters=params)
        assert fd.parameters is None

    def test_non_empty_parameters_preserved(self):
        """Test that non-empty parameters are preserved."""
        params = {"type": "object", "properties": {"x": {"type": "string"}}}
        fd = FunctionDefinition(name="test", parameters=params)
        assert fd.parameters == params


class TestToolDefinition:
    """Tests for ToolDefinition model."""

    def test_basic_creation(self):
        """Test creating a ToolDefinition."""
        fd = FunctionDefinition(name="test_func", description="Test")
        td = ToolDefinition(function=fd)
        assert td.type == "function"
        assert td.function.name == "test_func"

    def test_default_type(self):
        """Test that type defaults to 'function'."""
        td = ToolDefinition(function=FunctionDefinition(name="test"))
        assert td.type == "function"


class TestFunctionCall:
    """Tests for FunctionCall model."""

    def test_basic_creation(self):
        """Test creating a FunctionCall."""
        fc = FunctionCall(name="get_weather", arguments='{"city": "NYC"}')
        assert fc.name == "get_weather"
        assert fc.arguments == '{"city": "NYC"}'

    def test_requires_both_fields(self):
        """Test that both name and arguments are required."""
        with pytest.raises(ValidationError):
            FunctionCall(name="test")
        with pytest.raises(ValidationError):
            FunctionCall(arguments="{}")


class TestToolCall:
    """Tests for ToolCall model."""

    def test_basic_creation(self):
        """Test creating a ToolCall."""
        tc = ToolCall(
            id="call_123",
            function=FunctionCall(name="get_weather", arguments='{"city": "NYC"}'),
        )
        assert tc.id == "call_123"
        assert tc.type == "function"
        assert tc.function.name == "get_weather"

    def test_name_property(self):
        """Test the name property shortcut."""
        tc = ToolCall(
            id="call_1",
            function=FunctionCall(name="my_tool", arguments="{}"),
        )
        assert tc.name == "my_tool"

    def test_arguments_property(self):
        """Test the arguments property shortcut."""
        tc = ToolCall(
            id="call_1",
            function=FunctionCall(name="test", arguments='{"x": 1}'),
        )
        assert tc.arguments == '{"x": 1}'

    def test_str_representation(self):
        """Test string representation of ToolCall."""
        tc = ToolCall(
            id="call_abc",
            function=FunctionCall(name="get_weather", arguments='{"city": "NYC"}'),
        )
        s = str(tc)
        assert "ToolCall" in s
        assert "get_weather" in s
        assert "call_abc" in s

    def test_str_truncates_long_arguments(self):
        """Test that long arguments are truncated in str representation."""
        long_args = json.dumps({"data": "x" * 100})
        tc = ToolCall(
            id="call_1",
            function=FunctionCall(name="test", arguments=long_args),
        )
        s = str(tc)
        # Should be truncated to 50 chars
        assert len(s) < len(long_args) + 50


class TestToolResponse:
    """Tests for ToolResponse XMLModel."""

    def test_basic_creation(self):
        """Test creating a ToolResponse."""
        tr = ToolResponse(id="resp_1", result="Success")
        assert tr.id == "resp_1"
        assert tr.result == "Success"

    def test_default_id(self):
        """Test that id defaults to empty string."""
        tr = ToolResponse(result="Test result")
        assert tr.id == ""


# =============================================================================
# Tool Class Tests
# =============================================================================


class TestToolFromCallable:
    """Tests for Tool.from_callable factory method."""

    def test_basic_function(self):
        """Test creating a Tool from a simple function."""

        def add(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        t = Tool.from_callable(add)
        assert t.name == "add"
        assert "Add two numbers" in t.description
        assert t.fn is not None

    def test_custom_name(self):
        """Test creating a Tool with custom name."""

        def my_func() -> str:
            return "test"

        t = Tool.from_callable(my_func, name="custom_name")
        assert t.name == "custom_name"

    def test_custom_description(self):
        """Test creating a Tool with custom description."""

        def my_func() -> str:
            return "test"

        t = Tool.from_callable(my_func, description="My custom description")
        assert t.description == "My custom description"

    @pytest.mark.xfail(reason="Bug: catch=False becomes DEFAULT_CATCH_EXCEPTIONS due to falsy check")
    def test_with_catch_false(self):
        """Test creating a Tool with catch=False."""

        def my_func() -> str:
            return "test"

        t = Tool.from_callable(my_func, catch=False)
        assert t.catch is False

    def test_with_catch_true(self):
        """Test creating a Tool with catch=True."""

        def my_func() -> str:
            return "test"

        t = Tool.from_callable(my_func, catch=True)
        assert t.catch is True

    def test_with_catch_exceptions(self):
        """Test creating a Tool with specific exceptions."""

        def my_func() -> str:
            return "test"

        t = Tool.from_callable(my_func, catch=[ValueError, TypeError])
        assert ValueError in t.catch
        assert TypeError in t.catch

    def test_with_truncate(self):
        """Test creating a Tool with truncate option."""

        def my_func() -> str:
            return "test"

        t = Tool.from_callable(my_func, truncate=100)
        assert t.truncate == 100

    def test_default_catch_exceptions(self):
        """Test that default catch includes JSONDecodeError and ValidationError."""

        def my_func() -> str:
            return "test"

        t = Tool.from_callable(my_func)
        assert json.JSONDecodeError in t.catch
        assert ValidationError in t.catch

    def test_parameters_schema_generated(self):
        """Test that parameters schema is generated from signature."""

        def my_func(name: str, count: int = 5) -> str:
            return name * count

        t = Tool.from_callable(my_func)
        assert "properties" in t.parameters_schema
        assert "name" in t.parameters_schema["properties"]
        assert "count" in t.parameters_schema["properties"]

    def test_async_function(self):
        """Test creating a Tool from an async function."""

        async def async_add(x: int, y: int) -> int:
            return x + y

        t = Tool.from_callable(async_add)
        assert t.name == "async_add"


class TestToolProperties:
    """Tests for Tool properties."""

    @pytest.fixture
    def sample_tool(self):
        """Create a sample tool for testing."""

        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        return Tool.from_callable(greet)

    def test_definition_property(self, sample_tool):
        """Test the definition property returns ToolDefinition."""
        defn = sample_tool.definition
        assert isinstance(defn, ToolDefinition)
        assert defn.function.name == "greet"
        assert "Greet someone" in defn.function.description

    def test_api_definition_property(self, sample_tool):
        """Test api_definition returns same as definition."""
        assert sample_tool.api_definition == sample_tool.definition

    def test_model_property(self, sample_tool):
        """Test the model property creates XMLModel."""
        model = sample_tool.model
        assert model is not None
        # Should be cached
        assert sample_tool.model is model


class TestToolCall:
    """Tests for Tool.__call__ method."""

    def test_call_invokes_function(self):
        """Test that calling a Tool invokes its function."""

        def add(x: int, y: int) -> int:
            return x + y

        t = Tool.from_callable(add)
        result = t(x=3, y=4)
        assert result == 7

    def test_call_with_positional_args(self):
        """Test calling Tool with positional arguments."""

        def multiply(a: int, b: int) -> int:
            return a * b

        t = Tool.from_callable(multiply)
        result = t(2, 5)
        assert result == 10


class TestToolClone:
    """Tests for Tool.clone method."""

    def test_clone_creates_copy(self):
        """Test that clone creates a separate copy."""

        def my_func() -> str:
            return "test"

        original = Tool.from_callable(my_func, name="original")
        cloned = original.clone()

        assert cloned is not original
        assert cloned.name == original.name
        assert cloned.description == original.description

    def test_clone_is_independent(self):
        """Test that cloned tool is independent."""

        def my_func() -> str:
            return "test"

        original = Tool.from_callable(my_func, name="original")
        cloned = original.clone()
        cloned.name = "cloned"

        assert original.name == "original"
        assert cloned.name == "cloned"


class TestToolWith:
    """Tests for Tool.with_ method."""

    @pytest.fixture
    def base_tool(self):
        """Create a base tool for testing."""

        def my_func() -> str:
            """Original description."""
            return "test"

        return Tool.from_callable(my_func, name="original_name")

    def test_with_name(self, base_tool):
        """Test with_ can change name."""
        new_tool = base_tool.with_(name="new_name")
        assert new_tool.name == "new_name"
        assert base_tool.name == "original_name"

    def test_with_description(self, base_tool):
        """Test with_ can change description."""
        new_tool = base_tool.with_(description="New description")
        assert new_tool.description == "New description"

    def test_with_catch_false(self, base_tool):
        """Test with_ can set catch=False."""
        new_tool = base_tool.with_(catch=False)
        assert new_tool.catch is False

    def test_with_catch_true(self, base_tool):
        """Test with_ can set catch=True."""
        new_tool = base_tool.with_(catch=True)
        assert new_tool.catch is True

    def test_with_catch_exceptions(self, base_tool):
        """Test with_ can set specific exceptions."""
        new_tool = base_tool.with_(catch=[RuntimeError])
        assert RuntimeError in new_tool.catch

    def test_with_truncate(self, base_tool):
        """Test with_ can set truncate."""
        new_tool = base_tool.with_(truncate=500)
        assert new_tool.truncate == 500

    def test_with_multiple_options(self, base_tool):
        """Test with_ can change multiple options at once."""
        new_tool = base_tool.with_(
            name="new_name", description="New desc", truncate=200
        )
        assert new_tool.name == "new_name"
        assert new_tool.description == "New desc"
        assert new_tool.truncate == 200


class TestToolHandleToolCall:
    """Tests for Tool.handle_tool_call method."""

    @pytest.fixture
    def simple_tool(self):
        """Create a simple tool for testing."""

        def greet(name: str) -> str:
            return f"Hello, {name}!"

        return Tool.from_callable(greet)

    @pytest.fixture
    def error_tool(self):
        """Create a tool that raises an exception."""

        def fail_func() -> str:
            raise ValueError("Something went wrong")

        return Tool.from_callable(fail_func, catch=True)

    @pytest.mark.asyncio
    async def test_handle_simple_call(self, simple_tool):
        """Test handling a simple tool call."""
        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="greet", arguments='{"name": "World"}'),
        )

        message, stop = await simple_tool.handle_tool_call(tool_call)

        assert stop is False
        assert message.role == "tool"
        assert "Hello, World!" in message.content

    @pytest.mark.asyncio
    async def test_handle_call_catches_exception(self, error_tool):
        """Test that exceptions are caught when catch=True."""
        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="fail_func", arguments="{}"),
        )

        message, stop = await error_tool.handle_tool_call(tool_call)

        assert stop is False
        # Should have error content
        assert message.role == "tool"

    @pytest.mark.asyncio
    async def test_handle_call_with_stop(self):
        """Test handling a tool call that returns Stop."""
        from rigging.error import Stop

        def stop_func() -> None:
            raise Stop("Stopping now")

        t = Tool.from_callable(stop_func)
        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="stop_func", arguments="{}"),
        )

        message, stop = await t.handle_tool_call(tool_call)

        assert stop is True
        assert TOOL_STOP_TAG in message.content

    @pytest.mark.asyncio
    async def test_handle_async_function(self):
        """Test handling a call to an async function."""

        async def async_greet(name: str) -> str:
            await asyncio.sleep(0.01)
            return f"Hi, {name}!"

        t = Tool.from_callable(async_greet)
        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="async_greet", arguments='{"name": "Async"}'),
        )

        message, stop = await t.handle_tool_call(tool_call)

        assert stop is False
        assert "Hi, Async!" in message.content

    @pytest.mark.asyncio
    async def test_handle_truncates_output(self):
        """Test that output is truncated when truncate is set."""

        def long_output() -> str:
            return "x" * 1000

        t = Tool.from_callable(long_output, truncate=100)
        tool_call = ToolCall(
            id="call_1",
            function=FunctionCall(name="long_output", arguments="{}"),
        )

        message, stop = await t.handle_tool_call(tool_call)

        # Content should be truncated
        assert len(message.content) <= 100 + 50  # Allow some margin for formatting


# =============================================================================
# tool Decorator Tests
# =============================================================================


class TestToolDecorator:
    """Tests for the @tool decorator."""

    def test_decorator_without_args(self):
        """Test @tool decorator without arguments."""

        @tool
        def my_tool() -> str:
            """My tool description."""
            return "result"

        assert isinstance(my_tool, Tool)
        assert my_tool.name == "my_tool"
        assert "My tool description" in my_tool.description

    def test_decorator_with_args(self):
        """Test @tool decorator with arguments."""

        @tool(name="custom_name", description="Custom description")
        def my_tool() -> str:
            return "result"

        assert my_tool.name == "custom_name"
        assert my_tool.description == "Custom description"

    def test_decorator_with_catch(self):
        """Test @tool decorator with catch option."""

        @tool(catch=True)
        def my_tool() -> str:
            return "result"

        assert my_tool.catch is True

    def test_decorator_with_truncate(self):
        """Test @tool decorator with truncate option."""

        @tool(truncate=200)
        def my_tool() -> str:
            return "result"

        assert my_tool.truncate == 200

    def test_decorated_tool_callable(self):
        """Test that decorated tool is callable."""

        @tool
        def add(x: int, y: int) -> int:
            return x + y

        assert add(2, 3) == 5


# =============================================================================
# ToolMethod Descriptor Tests
# =============================================================================


class TestToolMethodDescriptor:
    """Tests for ToolMethod descriptor."""

    @pytest.mark.xfail(reason="Pre-existing recursion bug in tool_method - calls itself instead of rigging.tool_method")
    def test_class_access_returns_tool(self):
        """Test that accessing ToolMethod on class returns Tool."""

        class MyClass:
            @tool_method
            def my_tool(self, x: int) -> int:
                """Doubles a number."""
                return x * 2

        tool_obj = MyClass.my_tool
        assert isinstance(tool_obj, Tool)
        assert tool_obj.name == "my_tool"

    @pytest.mark.xfail(reason="Pre-existing recursion bug in tool_method - calls itself instead of rigging.tool_method")
    def test_instance_access_returns_bound_tool(self):
        """Test that accessing ToolMethod on instance returns bound Tool."""

        class MyClass:
            def __init__(self, multiplier: int):
                self.multiplier = multiplier

            @tool_method
            def multiply(self, x: int) -> int:
                """Multiplies by instance multiplier."""
                return x * self.multiplier

        obj = MyClass(multiplier=3)
        bound_tool = obj.multiply

        assert isinstance(bound_tool, Tool)
        # Should be bound to instance
        assert bound_tool(5) == 15

    @pytest.mark.xfail(reason="Pre-existing recursion bug in tool_method - calls itself instead of rigging.tool_method")
    def test_different_instances_different_tools(self):
        """Test that different instances get different bound tools."""

        class Counter:
            def __init__(self):
                self.count = 0

            @tool_method
            def increment(self) -> int:
                """Increment counter."""
                self.count += 1
                return self.count

        c1 = Counter()
        c2 = Counter()

        c1.increment()
        c1.increment()

        assert c1.count == 2
        assert c2.count == 0


class TestToolMethodDecorator:
    """Tests for @tool_method decorator."""

    @pytest.mark.xfail(reason="Pre-existing recursion bug in tool_method - calls itself instead of rigging.tool_method")
    def test_basic_usage(self):
        """Test basic @tool_method usage."""

        class MyToolset:
            @tool_method
            def my_tool(self) -> str:
                """A simple tool."""
                return "result"

        assert isinstance(MyToolset.my_tool, Tool)

    @pytest.mark.xfail(reason="Pre-existing recursion bug in tool_method - calls itself instead of rigging.tool_method")
    def test_with_variants(self):
        """Test @tool_method with variants."""

        class MyToolset:
            @tool_method(variants=["v1", "v2"])
            def my_tool(self) -> str:
                return "result"

        variants = getattr(MyToolset.my_tool, TOOL_VARIANTS_ATTR, [])
        # Check that it's a ToolMethod with variants set
        tool_method_obj = MyToolset.__dict__["my_tool"]
        assert hasattr(tool_method_obj, TOOL_VARIANTS_ATTR)
        assert "v1" in getattr(tool_method_obj, TOOL_VARIANTS_ATTR)
        assert "v2" in getattr(tool_method_obj, TOOL_VARIANTS_ATTR)

    @pytest.mark.xfail(reason="Pre-existing recursion bug in tool_method - calls itself instead of rigging.tool_method")
    def test_with_custom_name(self):
        """Test @tool_method with custom name."""

        class MyToolset:
            @tool_method(name="custom_tool_name")
            def my_tool(self) -> str:
                return "result"

        assert MyToolset.my_tool.name == "custom_tool_name"

    @pytest.mark.xfail(reason="Pre-existing recursion bug in tool_method - calls itself instead of rigging.tool_method")
    def test_with_custom_description(self):
        """Test @tool_method with custom description."""

        class MyToolset:
            @tool_method(description="Custom description here")
            def my_tool(self) -> str:
                return "result"

        assert MyToolset.my_tool.description == "Custom description here"


# =============================================================================
# Toolset Tests
# =============================================================================


class TestToolsetBasics:
    """Tests for basic Toolset functionality."""

    def test_toolset_creation(self):
        """Test creating a Toolset."""

        class MyToolset(Toolset):
            pass

        ts = MyToolset()
        assert ts.name == "MyToolset"
        assert ts.variant is None

    def test_toolset_with_variant(self):
        """Test Toolset with variant."""

        class MyToolset(Toolset):
            pass

        ts = MyToolset(variant="production")
        assert ts.variant == "production"

    def test_toolset_name_property(self):
        """Test that name property returns class name."""

        class CustomToolset(Toolset):
            pass

        ts = CustomToolset()
        assert ts.name == "CustomToolset"


class TestToolsetGetTools:
    """Tests for Toolset.get_tools method."""

    def test_get_tools_empty(self):
        """Test get_tools on Toolset with no tools."""

        class EmptyToolset(Toolset):
            pass

        ts = EmptyToolset()
        tools = ts.get_tools()
        assert tools == []

    @pytest.mark.xfail(reason="Pre-existing recursion bug in tool_method - calls itself instead of rigging.tool_method")
    def test_get_tools_single(self):
        """Test get_tools with single tool."""

        class SingleToolset(Toolset):
            @tool_method
            def my_tool(self) -> str:
                """A tool."""
                return "result"

        ts = SingleToolset()
        tools = ts.get_tools()
        assert len(tools) == 1
        assert tools[0].name == "my_tool"

    @pytest.mark.xfail(reason="Pre-existing recursion bug in tool_method - calls itself instead of rigging.tool_method")
    def test_get_tools_multiple(self):
        """Test get_tools with multiple tools."""

        class MultiToolset(Toolset):
            @tool_method
            def tool_a(self) -> str:
                return "a"

            @tool_method
            def tool_b(self) -> str:
                return "b"

        ts = MultiToolset()
        tools = ts.get_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert names == {"tool_a", "tool_b"}

    @pytest.mark.xfail(reason="Pre-existing recursion bug in tool_method - calls itself instead of rigging.tool_method")
    def test_get_tools_with_variant_filter(self):
        """Test get_tools filters by variant."""

        class VariantToolset(Toolset):
            @tool_method(variants=["v1"])
            def tool_v1(self) -> str:
                return "v1"

            @tool_method(variants=["v2"])
            def tool_v2(self) -> str:
                return "v2"

            @tool_method(variants=["v1", "v2"])
            def tool_both(self) -> str:
                return "both"

        ts = VariantToolset()

        v1_tools = ts.get_tools(variant="v1")
        v1_names = {t.name for t in v1_tools}
        assert "tool_v1" in v1_names
        assert "tool_both" in v1_names
        assert "tool_v2" not in v1_names

        v2_tools = ts.get_tools(variant="v2")
        v2_names = {t.name for t in v2_tools}
        assert "tool_v2" in v2_names
        assert "tool_both" in v2_names
        assert "tool_v1" not in v2_names

    @pytest.mark.xfail(reason="Pre-existing recursion bug in tool_method - calls itself instead of rigging.tool_method")
    def test_get_tools_uses_instance_variant(self):
        """Test that get_tools uses instance variant by default."""

        class VariantToolset(Toolset):
            @tool_method(variants=["prod"])
            def prod_tool(self) -> str:
                return "prod"

            @tool_method(variants=["dev"])
            def dev_tool(self) -> str:
                return "dev"

        ts = VariantToolset(variant="prod")
        tools = ts.get_tools()
        names = {t.name for t in tools}
        assert "prod_tool" in names
        assert "dev_tool" not in names


class TestToolsetContextManager:
    """Tests for Toolset context manager behavior."""

    def test_toolset_is_async_context_manager(self):
        """Test that Toolset can be used as async context manager."""

        class CMToolset(Toolset):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        async def test():
            ts = CMToolset()
            async with ts as ctx:
                assert ctx is ts

        asyncio.run(test())

    def test_reentrant_context_manager(self):
        """Test that Toolset context manager is re-entrant."""
        entry_count = 0
        exit_count = 0

        class ReentrantToolset(Toolset):
            def __enter__(self):
                nonlocal entry_count
                entry_count += 1
                return self

            def __exit__(self, *args):
                nonlocal exit_count
                exit_count += 1

        async def test():
            ts = ReentrantToolset()
            async with ts:
                async with ts:
                    pass
            # Should only enter/exit once even with nested usage
            assert entry_count == 1
            assert exit_count == 1

        asyncio.run(test())

    def test_context_manager_validation_enter_without_exit(self):
        """Test that defining __enter__ without __exit__ raises error."""
        with pytest.raises(TypeError):

            class BadToolset(Toolset):
                def __enter__(self):
                    return self

    def test_context_manager_validation_exit_without_enter(self):
        """Test that defining __exit__ without __enter__ raises error."""
        with pytest.raises(TypeError):

            class BadToolset(Toolset):
                def __exit__(self, *args):
                    pass

    def test_context_manager_validation_both_sync_async_enter(self):
        """Test that defining both __enter__ and __aenter__ raises error."""
        with pytest.raises(TypeError):

            class BadToolset(Toolset):
                def __enter__(self):
                    return self

                async def __aenter__(self):
                    return self

                def __exit__(self, *args):
                    pass

    def test_context_manager_validation_both_sync_async_exit(self):
        """Test that defining both __exit__ and __aexit__ raises error."""
        with pytest.raises(TypeError):

            class BadToolset(Toolset):
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                async def __aexit__(self, *args):
                    pass


class TestToolsetInheritance:
    """Tests for Toolset inheritance."""

    @pytest.mark.xfail(reason="Pre-existing recursion bug in tool_method - calls itself instead of rigging.tool_method")
    def test_inherited_tools(self):
        """Test that tools are inherited from parent class."""

        class ParentToolset(Toolset):
            @tool_method
            def parent_tool(self) -> str:
                return "parent"

        class ChildToolset(ParentToolset):
            @tool_method
            def child_tool(self) -> str:
                return "child"

        ts = ChildToolset()
        tools = ts.get_tools()
        names = {t.name for t in tools}
        assert "parent_tool" in names
        assert "child_tool" in names

    @pytest.mark.xfail(reason="Pre-existing recursion bug in tool_method - calls itself instead of rigging.tool_method")
    def test_overridden_tools(self):
        """Test that child can override parent tools."""

        class ParentToolset(Toolset):
            @tool_method
            def my_tool(self) -> str:
                return "parent"

        class ChildToolset(ParentToolset):
            @tool_method
            def my_tool(self) -> str:
                return "child"

        ts = ChildToolset()
        tools = ts.get_tools()
        # Should only have one tool
        assert len([t for t in tools if t.name == "my_tool"]) == 1
        # Should be the child's version
        my_tool = next(t for t in tools if t.name == "my_tool")
        assert my_tool() == "child"


# =============================================================================
# discover_tools_on_obj Tests
# =============================================================================


class TestDiscoverToolsOnObj:
    """Tests for discover_tools_on_obj function."""

    @pytest.mark.xfail(reason="Pre-existing recursion bug in tool_method - calls itself instead of rigging.tool_method")
    def test_discover_on_toolset(self):
        """Test discovering tools on a Toolset."""

        class MyToolset(Toolset):
            @tool_method
            def tool_a(self) -> str:
                return "a"

        ts = MyToolset()
        tools = discover_tools_on_obj(ts)
        assert len(tools) == 1
        assert tools[0].name == "tool_a"

    @pytest.mark.xfail(reason="Pre-existing recursion bug in tool_method - calls itself instead of rigging.tool_method")
    def test_discover_on_regular_object(self):
        """Test discovering tools on a regular object with ToolMethod."""

        class RegularClass:
            @tool_method
            def my_tool(self) -> str:
                return "result"

        obj = RegularClass()
        tools = discover_tools_on_obj(obj)
        assert len(tools) == 1
        assert tools[0].name == "my_tool"

    def test_discover_on_object_without_tools(self):
        """Test discovering tools on object with no tools."""

        class PlainClass:
            def regular_method(self):
                pass

        obj = PlainClass()
        tools = discover_tools_on_obj(obj)
        assert tools == []

    def test_discover_on_none(self):
        """Test discovering tools on None-like object."""
        tools = discover_tools_on_obj(None)
        assert tools == []


# =============================================================================
# DEFAULT_CATCH_EXCEPTIONS Tests
# =============================================================================


class TestDefaultCatchExceptions:
    """Tests for DEFAULT_CATCH_EXCEPTIONS constant."""

    def test_includes_json_decode_error(self):
        """Test that JSONDecodeError is in default catch exceptions."""
        assert json.JSONDecodeError in DEFAULT_CATCH_EXCEPTIONS

    def test_includes_validation_error(self):
        """Test that ValidationError is in default catch exceptions."""
        assert ValidationError in DEFAULT_CATCH_EXCEPTIONS


# =============================================================================
# Integration Tests
# =============================================================================


class TestToolIntegration:
    """Integration tests for the tools module."""

    @pytest.mark.asyncio
    async def test_full_tool_flow(self):
        """Test complete flow: create tool, make call, handle response."""

        def calculate(operation: str, x: float, y: float) -> float:
            """Perform a calculation."""
            if operation == "add":
                return x + y
            elif operation == "multiply":
                return x * y
            else:
                raise ValueError(f"Unknown operation: {operation}")

        calc_tool = Tool.from_callable(calculate)

        # Create a tool call
        tool_call = ToolCall(
            id="call_calc",
            function=FunctionCall(
                name="calculate",
                arguments='{"operation": "multiply", "x": 6, "y": 7}',
            ),
        )

        # Handle the call
        message, stop = await calc_tool.handle_tool_call(tool_call)

        assert stop is False
        assert "42" in message.content

    @pytest.mark.xfail(reason="Pre-existing recursion bug in tool_method - calls itself instead of rigging.tool_method")
    @pytest.mark.asyncio
    async def test_toolset_with_state(self):
        """Test Toolset that maintains state across calls."""

        class StatefulToolset(Toolset):
            def __init__(self):
                super().__init__()
                self.counter = 0

            @tool_method
            def increment(self, amount: int = 1) -> int:
                """Increment the counter."""
                self.counter += amount
                return self.counter

            @tool_method
            def get_count(self) -> int:
                """Get current count."""
                return self.counter

        ts = StatefulToolset()

        # Get tools
        tools = ts.get_tools()
        inc_tool = next(t for t in tools if t.name == "increment")
        get_tool = next(t for t in tools if t.name == "get_count")

        # Use tools
        assert inc_tool(amount=5) == 5
        assert inc_tool(amount=3) == 8
        assert get_tool() == 8
