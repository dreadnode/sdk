import json
import typing as t
from unittest.mock import MagicMock

import pytest
import rigging as rg
from pydantic import PrivateAttr, ValidationError
from rigging.generator.base import GeneratedMessage

from dreadnode.agent.agent import Agent, TaskAgent
from dreadnode.agent.error import MaxStepsError
from dreadnode.agent.events import AgentEnd, AgentEvent, AgentStalled, Reacted, ToolStart
from dreadnode.agent.hooks.base import retry_with_feedback
from dreadnode.agent.reactions import RetryWithFeedback
from dreadnode.agent.stop import never, tool_use
from dreadnode.agent.tools import AnyTool, Tool, Toolset, tool, tool_method
from dreadnode.meta import component

# Fixtures and helper classes/functions


@pytest.fixture
def simple_tool() -> AnyTool:
    """A simple tool for testing."""

    def get_weather(city: str) -> str:
        """Gets the weather for a city."""
        return f"The weather in {city} is sunny."

    return Tool.from_callable(component(get_weather))


class MyToolset(Toolset):
    """A simple toolset for testing."""

    @tool_method
    def get_time(self) -> str:
        """Gets the current time."""
        return "The time is 12:00 PM."


class MockGenerator(rg.Generator):
    """
    A mock generator that returns a sequence of predefined responses.
    This allows for deterministic, stateful testing of agent conversations.
    """

    _responses: list[GeneratedMessage | Exception] = PrivateAttr(default_factory=list)
    _call_history: list[tuple[t.Sequence[rg.Message], rg.GenerateParams]] = PrivateAttr(
        default_factory=list
    )

    async def generate_messages(
        self,
        messages: t.Sequence[t.Sequence[rg.Message]],
        params: t.Sequence[rg.GenerateParams],
    ) -> t.Sequence[GeneratedMessage]:
        # Assume batch size of 1 for test simplicity
        self._call_history.append((messages[0], params[0]))

        if not self._responses:
            raise AssertionError("MockGenerator ran out of responses to provide.")

        next_response = self._responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response

        return [next_response]

    async def supports_function_calling(self) -> bool:
        return True

    @staticmethod
    def tool_response(
        tool_name: str, tool_args: dict[str, t.Any], tool_id: str = "tool_123"
    ) -> GeneratedMessage:
        """Helper to create a GeneratedMessage with a tool call."""
        return GeneratedMessage(
            message=rg.Message(
                role="assistant",
                content=None,
                tool_calls=[
                    rg.tools.ToolCall(
                        id=tool_id,
                        function=rg.tools.FunctionCall(
                            name=tool_name, arguments=json.dumps(tool_args)
                        ),
                    )
                ],
            ),
            stop_reason="tool_calls",
        )

    @staticmethod
    def text_response(content: str) -> GeneratedMessage:
        """Helper to create a simple text-based GeneratedMessage."""
        return GeneratedMessage(
            message=rg.Message(role="assistant", content=content),
            stop_reason="stop",
        )


@pytest.fixture
def mock_generator() -> MockGenerator:
    """Provides a fresh instance of the MockGenerator for each test."""
    return MockGenerator(model="mock-model", params=rg.GenerateParams(), api_key="test-key")


@pytest.fixture
def base_agent() -> Agent:
    """Provides a basic agent with a mock generator for reuse in tests."""
    return Agent(
        name="TestAgent",
        model=MockGenerator(model="mock-model", params=rg.GenerateParams(), api_key="test-key"),
    )


# Initialization and Configuration


def test_agent_initialization(mock_generator: MockGenerator) -> None:
    """Verify that a basic Agent can be created with correct attributes."""
    agent = Agent(
        name="MyAgent",
        description="A test agent.",
        model=mock_generator,
        instructions="Be helpful.",
    )

    assert agent.name == "MyAgent"
    assert agent.description == "A test agent."
    assert agent.instructions == "Be helpful."
    assert agent.generator is mock_generator


def test_tool_validation_and_discovery(simple_tool: AnyTool) -> None:
    """Test that the `tools` field correctly validates and discovers tools."""

    # 1. Test with a raw callable
    @tool
    def raw_func() -> None: ...

    agent_raw = Agent(name="test", tools=[raw_func])
    assert len(agent_raw.tools) == 1
    assert isinstance(agent_raw.tools[0], Tool)
    assert agent_raw.tools[0].name == "raw_func"

    # 2. Test with a pre-made Tool instance
    agent_tool = Agent(name="test", tools=[simple_tool])
    assert agent_tool.tools[0] is simple_tool

    # 3. Test with a Toolset instance
    toolset = MyToolset()
    agent_toolset = Agent(name="test", tools=[toolset])
    assert len(agent_toolset.all_tools) == 1
    assert agent_toolset.all_tools[0].name == "get_time"
    # Ensure the tool is bound to the instance
    assert agent_toolset.all_tools[0].fn.__self__ is toolset  # type: ignore[attr-defined]


def test_all_tools_property(simple_tool: AnyTool) -> None:
    """Ensure the .all_tools property correctly flattens tools from all sources."""
    toolset = MyToolset()
    agent = Agent(name="test", tools=[simple_tool, toolset])

    all_tools = agent.all_tools
    assert len(all_tools) == 2

    tool_names = {t.name for t in all_tools}
    assert "get_weather" in tool_names
    assert "get_time" in tool_names


def test_generator_property() -> None:
    """Verify the .generator property initializes from a string or instance."""
    generator_instance = MockGenerator(
        model="mock-model", params=rg.GenerateParams(), api_key="test-key"
    )
    agent_instance = Agent(name="test", model=generator_instance)
    assert agent_instance.generator is generator_instance

    with pytest.raises(ValidationError):
        Agent(name="test", model=123)  # type: ignore[arg-type]


# Fluent Configuration (`with_` method)


def test_with_method_is_immutable(base_agent: Agent, simple_tool: AnyTool) -> None:
    """Confirm that agent.with_() creates a new instance and does not mutate the original."""
    new_agent = base_agent.with_(name="NewName", tools=[simple_tool])

    assert new_agent is not base_agent
    assert base_agent.name == "TestAgent"
    assert new_agent.name == "NewName"
    assert len(base_agent.tools) == 0
    assert len(new_agent.tools) == 1


def test_with_method_replaces_attributes(base_agent: Agent, simple_tool: AnyTool) -> None:
    """Verify that calling .with_() replaces specified attributes."""
    new_agent = base_agent.with_(
        name="ReplacedAgent",
        max_steps=99,
        tools=[simple_tool],
        hooks=[MagicMock()],
    )
    assert new_agent.name == "ReplacedAgent"
    assert new_agent.max_steps == 99
    assert len(new_agent.tools) == 1
    assert new_agent.tools[0].name == "get_weather"
    assert len(new_agent.hooks) == 1


def test_with_method_replaces_lists_by_default(base_agent: Agent, simple_tool: AnyTool) -> None:
    """Verify that list-based attributes are fully replaced when append=False."""
    tool1 = simple_tool.model_copy(update={"name": "tool1"})
    tool2 = simple_tool.model_copy(update={"name": "tool2"})

    base_agent.tools = [tool1]
    new_agent = base_agent.with_(tools=[tool2])  # append=False is the default

    assert len(new_agent.all_tools) == 1
    assert new_agent.all_tools[0].name == "tool2"


def test_with_method_appends_lists(base_agent: Agent, simple_tool: AnyTool) -> None:
    """Verify that list-based attributes are appended to when append=True."""
    tool1 = simple_tool.model_copy(update={"name": "tool1"})
    tool2 = simple_tool.model_copy(update={"name": "tool2"})

    base_agent.tools = [tool1]
    new_agent = base_agent.with_(tools=[tool2], append=True)

    assert len(new_agent.all_tools) == 2
    tool_names = {t.name for t in new_agent.all_tools}
    assert tool_names == {"tool1", "tool2"}


# Core Execution Logic


@pytest.mark.asyncio
async def test_run_happy_path_single_tool_call(
    mock_generator: MockGenerator, simple_tool: AnyTool
) -> None:
    """
    Test a successful run where the agent calls one tool and then provides a final answer.
    """
    # 1. LLM asks to use the 'get_weather' tool.
    # 2. LLM provides the final answer based on the tool's output.
    mock_generator._responses = [
        MockGenerator.tool_response("get_weather", {"city": "London"}),
        MockGenerator.text_response("The weather in London is sunny."),
    ]

    agent = Agent(name="WeatherAgent", model=mock_generator, tools=[simple_tool])
    result = await agent.run("What's the weather in London?")

    assert not result.failed
    assert result.error is None
    assert result.steps == 2  # Step 1: Tool call, Step 2: Final answer
    assert result.stop_reason == "finished"

    # Verify the conversation history is correct
    messages = result.messages
    assert len(messages) == 4  # user -> assistant (tool) -> tool (result) -> assistant (final)
    assert messages[0].role == "user"
    assert messages[0].content == "What's the weather in London?"
    assert messages[1].role == "assistant"
    assert messages[1].tool_calls is not None
    assert messages[2].role == "tool"
    assert "The weather in London is sunny." in messages[2].content
    assert messages[3].role == "assistant"
    assert messages[3].content == "The weather in London is sunny."


@pytest.mark.asyncio
async def test_run_stops_on_max_steps(mock_generator: MockGenerator, simple_tool: AnyTool) -> None:
    """Ensure the agent run terminates with a MaxStepsError when exceeding max_steps."""
    # The agent will just keep calling the tool.
    mock_generator._responses = [
        MockGenerator.tool_response("get_weather", {"city": "A"}),
        MockGenerator.tool_response("get_weather", {"city": "B"}),
    ]

    agent = Agent(
        name="MaxStepsAgent",
        model=mock_generator,
        tools=[simple_tool],
        max_steps=1,
    )
    result = await agent.run("...")

    assert result.failed
    assert result.stop_reason == "max_steps_reached"
    assert isinstance(result.error, MaxStepsError)
    assert result.steps == 1


@pytest.mark.asyncio
async def test_run_stops_on_stop_condition(
    mock_generator: MockGenerator, simple_tool: AnyTool
) -> None:
    """Verify the agent stops successfully when a StopCondition is met."""
    from dreadnode.agent.stop import tool_use

    mock_generator._responses = [
        MockGenerator.tool_response("get_weather", {"city": "London"}),
        # This second response should never be used.
        MockGenerator.text_response("This should not be generated."),
    ]

    agent = Agent(
        name="StopConditionAgent",
        model=mock_generator,
        tools=[simple_tool],
        stop_conditions=[tool_use("get_weather")],
    )
    result = await agent.run("...")

    assert not result.failed
    assert result.stop_reason == "finished"
    assert result.steps == 1  # Stops after the tool call in step 1.

    # Only one call to the generator should have been made.
    assert len(mock_generator._call_history) == 1


@pytest.mark.asyncio
async def test_run_handles_stalling(mock_generator: MockGenerator) -> None:
    """
    Check for an AgentStalled event when the model produces no tool calls
    and no stop condition is met.
    """
    mock_generator._responses = [
        MockGenerator.text_response("I'm not sure what to do."),
        MockGenerator.text_response("I'm still not sure."),
    ]

    # A stop condition is required to trigger stalling logic
    agent = Agent(name="StallAgent", model=mock_generator, stop_conditions=[never()], max_steps=2)

    events = []
    async with agent.stream("...") as stream:
        async for event in stream:
            events.append(event)

    assert any(isinstance(e, AgentStalled) for e in events)

    # The agent should eventually fail due to max_steps
    final_event = events[-1]
    assert isinstance(final_event, AgentEnd)
    assert final_event.result.failed
    assert final_event.result.stop_reason == "stalled"


@pytest.mark.asyncio
async def test_run_fails_on_generation_error(mock_generator: MockGenerator) -> None:
    """Test that the agent fails gracefully if the generator raises an exception."""
    gen_error = RuntimeError("The API is down.")
    mock_generator._responses = [gen_error]

    agent = Agent(name="FailAgent", model=mock_generator)
    result = await agent.run("...")

    assert result.failed
    assert result.stop_reason == "error"
    assert result.error is gen_error


@pytest.mark.asyncio
async def test_run_handles_tool_execution_error(mock_generator: MockGenerator) -> None:
    """Verify the agent continues after a tool fails, adding an error message to history."""

    @tool(catch=True)
    def faulty_tool() -> None:
        raise ValueError("This tool is broken!")

    mock_generator._responses = [
        MockGenerator.tool_response("faulty_tool", {}),
        MockGenerator.text_response("Okay, the tool failed. I will stop."),
    ]

    agent = Agent(
        name="ToolFailAgent",
        model=mock_generator,
        tools=[faulty_tool],
    )
    result = await agent.run("...")

    assert not result.failed
    assert result.stop_reason == "finished"

    # Check that the error was captured in the tool message
    tool_message = result.messages[2]
    assert tool_message.role == "tool"
    assert "ValueError" in tool_message.content
    assert "This tool is broken!" in tool_message.content


# Hooks and Reactions


@pytest.mark.asyncio
async def test_hook_triggers_finish_reaction(
    mock_generator: MockGenerator, simple_tool: AnyTool
) -> None:
    """A hook returning Finish() should stop the agent successfully."""
    from dreadnode.agent.events import ToolStart
    from dreadnode.agent.reactions import Finish

    mock_generator._responses = [
        MockGenerator.tool_response("get_weather", {"city": "A"}),
    ]

    # This hook will fire on the first tool call and stop the run.
    async def stop_on_tool_start(event: t.Any) -> Finish | None:
        if isinstance(event, ToolStart):
            return Finish(reason="Stopped by hook")
        return None

    agent = Agent(
        name="FinishHookAgent",
        model=mock_generator,
        tools=[simple_tool],
        hooks=[stop_on_tool_start],
    )
    result = await agent.run("...")

    assert not result.failed
    assert result.stop_reason == "finished"
    assert len(mock_generator._call_history) == 1


@pytest.mark.asyncio
async def test_hook_triggers_retry_with_feedback_reaction(
    mock_generator: MockGenerator, simple_tool: AnyTool
) -> None:
    """A hook returning RetryWithFeedback should inject a message and re-run generation."""
    mock_generator._responses = [
        MockGenerator.text_response("I am stuck."),
        MockGenerator.tool_response("get_weather", {"city": "B"}),
    ]

    agent = Agent(
        name="FeedbackAgent",
        model=mock_generator,
        tools=[simple_tool],
        stop_conditions=[tool_use("get_weather")],  # Force stalling
        hooks=[
            retry_with_feedback(
                event_type=AgentStalled,
                feedback="You must use a tool to proceed.",
            )
        ],
    )
    result = await agent.run("...")

    assert not result.failed
    assert len(mock_generator._call_history) == 2

    # Check the messages sent to the generator on the second call.
    # It should include the feedback from the hook.
    messages_for_second_call = mock_generator._call_history[1][0]
    last_message = messages_for_second_call[-1]
    assert last_message.role == "user"
    assert last_message.content == "You must use a tool to proceed."


@pytest.mark.asyncio
async def test_reaction_handling_and_event_sequence(
    mock_generator: MockGenerator, simple_tool: AnyTool
) -> None:
    """
    Tests the precise sequence of events when a hook triggers a reaction.

    It uses two hooks:
    1. A 'trigger' hook that fires a RetryWithFeedback reaction on ToolStart.
    2. An 'observer' hook that records all events it sees.

    The test verifies that the observer sees the ToolStart event, followed
    immediately by the Reacted event, and that the agent correctly acts on
    the reaction by retrying with the new feedback message.
    """

    # --- Test-specific Hooks ---
    async def trigger_hook(event: t.Any) -> RetryWithFeedback | None:
        """If this hook sees a tool start, it fires a reaction."""
        if isinstance(event, ToolStart):
            return RetryWithFeedback(feedback="The hook reacted!")
        return None

    class ObserverHook:
        """A stateful hook to record the sequence of events."""

        def __init__(self) -> None:
            self.seen_events: list[AgentEvent] = []

        async def __call__(self, event: t.Any) -> None:
            self.seen_events.append(event)

    # --- Test Setup ---
    observer = ObserverHook()

    # 1. First generation leads to a tool call, which will be interrupted by the hook.
    # 2. Second generation (after the retry) leads to a final text response.
    mock_generator._responses = [
        MockGenerator.tool_response("get_weather", {"city": "Test"}),
        MockGenerator.text_response("Okay, I have been reset."),
    ]

    agent = Agent(
        name="ReactionTestAgent",
        model=mock_generator,
        tools=[simple_tool],
        hooks=[trigger_hook, observer],  # Both hooks are active
    )
    await agent.run("...")

    # --- Assertions ---

    # 1. Verify the correct events were seen in the correct order.
    event_types = [type(e).__name__ for e in observer.seen_events]

    try:
        tool_start_index = event_types.index("ToolStart")
    except ValueError:
        pytest.fail("ObserverHook never saw a ToolStart event.")

    # The 'Reacted' event must immediately follow the event that caused it.
    assert event_types[tool_start_index + 1] == "Reacted"

    # 2. Introspect the Reacted event to ensure it's correct.
    reacted_event = observer.seen_events[tool_start_index + 1]
    assert isinstance(reacted_event, Reacted)

    # The hook name might be slightly different based on environment, but should be 'trigger_hook'
    assert reacted_event.hook_name == "trigger_hook"
    assert isinstance(reacted_event.reaction, RetryWithFeedback)
    assert reacted_event.reaction.feedback == "The hook reacted!"

    # 3. Verify the agent actually performed the retry.
    assert len(mock_generator._call_history) == 2, "Generator should have been called twice."

    # 4. Verify the feedback from the reaction was in the messages for the second call.
    messages_for_second_call = mock_generator._call_history[1][0]
    feedback_message = messages_for_second_call[-1]
    assert feedback_message.role == "user"
    assert feedback_message.content == "The hook reacted!"


# Quality of Life and Edge Cases


def test_instructions_and_description_are_dedented() -> None:
    """Verify that multiline, indented strings are dedented on initialization."""
    indented_str = """\
        This is a multi-line,
        indented string.
        It should be cleaned up.
    """
    expected = "This is a multi-line,\nindented string.\nIt should be cleaned up.\n"

    agent = Agent(
        name="DedentAgent",
        description=indented_str,
        instructions=indented_str,
        model="test-model",
    )

    assert agent.description == expected
    assert agent.instructions == expected

    # Edge case: None should pass through without error
    agent_none = Agent(name="NoneAgent", instructions=None, model="test-model")
    assert agent_none.instructions is None


def test_tool_validator_handles_nested_lists_and_plain_objects(simple_tool: AnyTool) -> None:
    """Verify the tool validator can flatten nested lists and discover tools."""

    toolset = MyToolset()

    # The validator should handle this messy, nested structure gracefully.
    agent = Agent(
        name="ToolValidatorTest",
        model="test-model",
        tools=[[simple_tool, toolset], toolset],  # type: ignore[list-item]
    )

    all_tools = agent.all_tools
    assert len(all_tools) == 3
    tool_names = {t.name for t in all_tools}
    assert tool_names == {"get_weather", "get_time"}


def test_with_method_retriggers_model_post_init_on_subclasses() -> None:
    """
    Verify that agent.with_() correctly re-runs model_post_init on subclasses,
    ensuring specialized agent logic is preserved.
    """
    # TaskAgent's model_post_init adds 3 tools, 1 hook, and 1 stop condition.
    agent = TaskAgent(name="MyTaskAgent", model="test-model")

    # The base agent should have the defaults from TaskAgent
    assert len(agent.all_tools) == 3
    assert any(t.name == "finish_task" for t in agent.all_tools)
    assert len(agent.hooks) == 1
    assert len(agent.stop_conditions) == 1
    assert isinstance(agent.stop_conditions[0], type(never()))

    # Create a new agent using `with_`, which should re-trigger model_post_init
    new_agent = agent.with_(name="NewTaskAgentName")

    # The new agent should also have the defaults, proving the hook ran again.
    assert len(new_agent.all_tools) == 3
    assert any(t.name == "finish_task" for t in new_agent.all_tools)
    assert len(new_agent.hooks) == 1
    assert len(new_agent.stop_conditions) == 1
    assert isinstance(new_agent.stop_conditions[0], type(never()))

    # Let's test idempotency: if we provide a tool that would be added, it shouldn't be duplicated
    from dreadnode.agent.tools.tasking import finish_task

    idempotent_agent = TaskAgent(name="Idempotent", model="test-model", tools=[finish_task])
    # The length should still be 3, not 4.
    assert len(idempotent_agent.all_tools) == 3
    assert sum(1 for t in idempotent_agent.all_tools if t.name == "finish_task") == 1
