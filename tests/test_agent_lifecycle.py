import asyncio
import inspect
import typing as t

import pytest

from dreadnode.agent import Agent
from dreadnode.agent.tools import Toolset, tool, tool_method

if t.TYPE_CHECKING:
    from dreadnode.agent.tools.base import AnyTool

# This is the state tracker that will record the order of events.
event_log: list[str] = []


class AsyncCMToolSet(Toolset):
    """
    Scenario 1: A standard, async-native Toolset.
    Tests that __aenter__/__aexit__ are called correctly and only once.
    """

    enter_count: int = 0
    exit_count: int = 0

    async def __aenter__(self) -> "AsyncCMToolSet":
        event_log.append("async_tool_enter_start")
        await asyncio.sleep(0.01)  # Simulate async work
        self.enter_count += 1
        event_log.append("async_tool_enter_end")
        return self

    async def __aexit__(self, *args: object) -> None:
        event_log.append("async_tool_exit_start")
        await asyncio.sleep(0.01)
        self.exit_count += 1
        event_log.append("async_tool_exit_end")

    @tool_method
    async def do_work(self) -> str:
        """A sample method for the agent to call."""
        event_log.append("async_tool_method_called")
        return "async work done"


class SyncCMToolSet(Toolset):
    """
    Scenario 2: A Toolset using synchronous __enter__/__exit__.
    Tests that our magic bridge correctly calls them in order.
    """

    def __enter__(self) -> "SyncCMToolSet":
        event_log.append("sync_tool_enter")
        return self

    def __exit__(self, *args: object) -> None:
        event_log.append("sync_tool_exit")

    @tool_method
    def do_blocking_work(self) -> str:
        """A sample sync method."""
        event_log.append("sync_tool_method_called")
        return "sync work done"


@tool
async def stateless_tool() -> str:
    """
    Scenario 3: A simple, stateless tool.
    Tests that the lifecycle manager ignores it.
    """
    event_log.append("stateless_tool_called")
    return "stateless work done"


class StandaloneCMToolset(Toolset):
    """
    Scenario 4 (Revised): A Toolset that acts as a standalone context manager.
    It must inherit from Toolset to be preserved by the Agent's validator.
    """

    async def __aenter__(self) -> "StandaloneCMToolset":
        event_log.append("standalone_cm_enter")
        return self

    async def __aexit__(self, *args: object) -> None:
        event_log.append("standalone_cm_exit")

    @tool_method
    async def do_standalone_work(self) -> str:
        """The callable part of the tool."""
        event_log.append("standalone_cm_called")
        return "standalone work done"


class ReturnValueToolSet(Toolset):
    """
    Scenario 5: A Toolset whose __aenter__ returns a different object.
    Tests that the `as` clause contract is honored.
    """

    class Handle:
        def __init__(self, message: str) -> None:
            self.message = message

    async def __aenter__(self) -> "ReturnValueToolSet.Handle":
        event_log.append("return_value_tool_enter")
        # Return a handle object, NOT self
        return self.Handle("special handle")

    async def __aexit__(self, *args: object) -> None:
        event_log.append("return_value_tool_exit")


# --- Mock Agent to Control Execution ---


class MockAgent(Agent):
    """
    An agent override that doesn't call an LLM. Instead, it simulates
    a run where it calls every available tool once.
    """

    async def _stream_traced(  # type: ignore[override]
        self,
        thread: object,  # noqa: ARG002
        user_input: str,  # noqa: ARG002
        *,
        commit: bool = True,  # noqa: ARG002
    ) -> t.AsyncIterator[str]:
        event_log.append("agent_run_start")
        # Simulate calling each tool the agent knows about
        for tool_ in self.all_tools:
            result = tool_()
            if inspect.isawaitable(result):
                await result
        event_log.append("agent_run_end")
        # Yield a dummy event to satisfy the stream consumer
        yield "dummy_event"


# --- The Tests ---


@pytest.mark.asyncio
async def test_agent_manages_all_lifecycle_scenarios() -> None:
    """
    Main integration test. Verifies that the Agent correctly manages setup,
    execution, and teardown for a mix of tool types in the correct order.
    """
    event_log.clear()

    # 1. Setup our collection of tools
    async_tool = AsyncCMToolSet()
    sync_tool = SyncCMToolSet()
    standalone_toolset = StandaloneCMToolset()

    # The list passed to the Agent contains the containers
    agent_tools: list[AnyTool | Toolset] = [
        async_tool,
        sync_tool,
        stateless_tool,
        standalone_toolset,
    ]

    agent = MockAgent(name="test_agent", tools=agent_tools)

    # 2. Execute the agent run within its stream context
    async with agent.stream("test input") as stream:
        event_log.append("stream_context_active")
        async for _ in stream:
            pass  # Consume the stream to trigger the run

    # 3. Assert the order of events
    expected_order = [
        # Setup phase (order of entry is guaranteed by list order)
        "async_tool_enter_start",
        "async_tool_enter_end",
        "sync_tool_enter",
        "standalone_cm_enter",
        # Agent execution phase
        "stream_context_active",
        "agent_run_start",
        "agent_run_end",
        # Teardown phase (must be LIFO)
        "standalone_cm_exit",
        "sync_tool_exit",
        "async_tool_exit_start",
        "async_tool_exit_end",
    ]

    # Extract the tool call events to check for presence separately
    run_events = [e for e in event_log if e.endswith("_called")]
    actual_order_without_run_events = [e for e in event_log if not e.endswith("_called")]

    assert actual_order_without_run_events == expected_order
    assert sorted(run_events) == sorted(
        [
            "async_tool_method_called",
            "sync_tool_method_called",
            "stateless_tool_called",
            "standalone_cm_called",
        ]
    )

    # 4. Assert idempotency (enter/exit should have been called only once)
    assert async_tool.enter_count == 1
    assert async_tool.exit_count == 1


@pytest.mark.asyncio
async def test_toolset_idempotency_wrapper() -> None:
    """
    A tight unit test to verify that our wrapper magic correctly
    prevents a toolset from being entered more than once.
    """
    tool = AsyncCMToolSet()

    # Nesting the context manager simulates the agent entering it
    # after the user might have manually (and incorrectly) entered it.
    async with tool as outer_handle:
        assert tool.enter_count == 1
        async with tool as inner_handle:
            assert tool.enter_count == 1  # Should NOT have increased
            assert inner_handle is outer_handle  # Should return the same handle

    assert tool.exit_count == 1  # Exit logic should only have run once


@pytest.mark.asyncio
async def test_toolset_return_value_is_honored() -> None:
    """
    Verifies that the handle returned by a custom __aenter__ is preserved.
    """
    tool = ReturnValueToolSet()

    async with tool as handle:
        assert isinstance(handle, tool.Handle)
        assert handle.message == "special handle"
