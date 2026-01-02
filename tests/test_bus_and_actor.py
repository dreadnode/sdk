"""
Tests for MessageBus and AgentActor components.

These tests verify the pub/sub message bus and the Ray Actor base classes.
"""

import asyncio
import pytest
from dataclasses import dataclass

from dreadnode.core.agents.bus import (
    InMemoryMessageBus,
    MessageBusConfig,
    create_message_bus,
)
from dreadnode.core.agents.events import WorkflowEventBase
from dreadnode.core.agents.orchestrator import ActorBase


# =============================================================================
# Test Events
# =============================================================================


class SampleEvent(WorkflowEventBase):
    """Simple sample event for testing."""
    topic: str = "test.event"
    task_id: str = "test-task-1"
    data: str = ""


class ResultEvent(WorkflowEventBase):
    """Result event for testing."""
    topic: str = "test.result"
    task_id: str = "test-task-1"
    result: str = ""


# =============================================================================
# InMemoryMessageBus Tests
# =============================================================================


class TestInMemoryMessageBus:
    """Tests for InMemoryMessageBus."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test basic start/stop lifecycle."""
        config = MessageBusConfig(
            producer_topics=["test.topic"],
            consumer_topics=["test.topic"],
        )
        bus = InMemoryMessageBus(config)

        await bus.start()
        assert bus._running is True
        assert "test.topic" in bus._queues

        await bus.stop()
        assert bus._running is False

    @pytest.mark.asyncio
    async def test_publish_creates_queue(self):
        """Test that publishing creates queue if not exists."""
        bus = InMemoryMessageBus()
        await bus.start()

        event = SampleEvent(data="hello")
        await bus.publish("new.topic", event)

        assert "new.topic" in bus._queues
        await bus.stop()

    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        """Test basic pub/sub flow."""
        config = MessageBusConfig(
            producer_topics=["test.events"],
            consumer_topics=["test.events"],
        )
        bus = InMemoryMessageBus(config)
        await bus.start()

        # Publish event
        event = SampleEvent(data="test message")
        await bus.publish("test.events", event)

        # Subscribe and receive
        received = []

        async def consume():
            async for evt in bus.subscribe(["test.events"]):
                received.append(evt)
                break  # Just get one

        # Run consumer with timeout
        try:
            await asyncio.wait_for(consume(), timeout=1.0)
        except asyncio.TimeoutError:
            pass

        assert len(received) == 1
        assert received[0].data == "test message"

        await bus.stop()

    @pytest.mark.asyncio
    async def test_multiple_topics(self):
        """Test subscribing to multiple topics."""
        config = MessageBusConfig(
            producer_topics=["topic.a", "topic.b"],
            consumer_topics=["topic.a", "topic.b"],
        )
        bus = InMemoryMessageBus(config)
        await bus.start()

        # Publish to different topics
        await bus.publish("topic.a", SampleEvent(data="from A"))
        await bus.publish("topic.b", SampleEvent(data="from B"))

        # Subscribe to both
        received = []

        async def consume():
            async for evt in bus.subscribe(["topic.a", "topic.b"]):
                received.append(evt)
                if len(received) >= 2:
                    break

        try:
            await asyncio.wait_for(consume(), timeout=2.0)
        except asyncio.TimeoutError:
            pass

        assert len(received) == 2
        assert {r.data for r in received} == {"from A", "from B"}

        await bus.stop()

    @pytest.mark.asyncio
    async def test_publish_and_wait(self):
        """Test publish_and_wait (same as publish for in-memory)."""
        bus = InMemoryMessageBus()
        await bus.start()

        event = SampleEvent(data="sync publish")
        await bus.publish_and_wait("test.topic", event)

        # Should be in queue
        assert not bus._queues["test.topic"].empty()

        await bus.stop()


# =============================================================================
# RayQueueMessageBus Tests
# =============================================================================

# Check if Ray is available
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


@pytest.fixture(scope="class")
def ray_context():
    """Initialize Ray for tests."""
    if not RAY_AVAILABLE:
        pytest.skip("Ray not available")

    # Shutdown any existing Ray context to ensure clean state
    if ray.is_initialized():
        ray.shutdown()

    # Initialize with local mode for testing
    ray.init(
        ignore_reinit_error=True,
        num_cpus=2,
        include_dashboard=False,
        _temp_dir="/tmp/ray_test",
    )

    yield

    # Shutdown after tests
    ray.shutdown()


@pytest.mark.skipif(not RAY_AVAILABLE, reason="Ray not available")
class TestRayQueueMessageBus:
    """Tests for RayQueueMessageBus.

    Note: Tests that involve actual Ray Queue operations require a properly
    configured Ray environment where all workers use the same Python version.
    These are marked with @pytest.mark.ray_integration.
    """

    @pytest.mark.asyncio
    async def test_start_stop(self, ray_context):
        """Test basic start/stop lifecycle."""
        from dreadnode.core.agents.bus import RayQueueMessageBus

        config = MessageBusConfig(
            producer_topics=["ray.test.topic"],
            consumer_topics=["ray.test.topic"],
        )
        bus = RayQueueMessageBus(config)

        await bus.start()
        assert bus._running is True
        assert "ray.test.topic" in bus._queues

        await bus.stop()
        assert bus._running is False

    @pytest.mark.asyncio
    async def test_queue_max_size(self, ray_context):
        """Test queue respects max size config."""
        from dreadnode.core.agents.bus import RayQueueMessageBus

        config = MessageBusConfig(
            producer_topics=["ray.sized"],
            queue_max_size=5,
        )
        bus = RayQueueMessageBus(config)
        await bus.start()

        # The queue should be created with the specified max size
        assert "ray.sized" in bus._queues

        await bus.stop()

    @pytest.mark.asyncio
    async def test_factory_creates_ray_bus(self, ray_context):
        """Test factory creates RayQueueMessageBus."""
        from dreadnode.core.agents.bus import RayQueueMessageBus

        config = MessageBusConfig()
        bus = create_message_bus("ray", config)

        assert isinstance(bus, RayQueueMessageBus)

    # Integration tests that require a properly configured Ray cluster
    # Run with: pytest -m ray_integration

    @pytest.mark.ray_integration
    @pytest.mark.asyncio
    async def test_publish_creates_queue(self, ray_context):
        """Test that publishing creates queue if not exists."""
        from dreadnode.core.agents.bus import RayQueueMessageBus

        bus = RayQueueMessageBus(MessageBusConfig())
        await bus.start()

        event = SampleEvent(data="ray hello")
        await bus.publish("ray.new.topic", event)

        assert "ray.new.topic" in bus._queues
        await bus.stop()

    @pytest.mark.ray_integration
    @pytest.mark.asyncio
    async def test_publish_subscribe(self, ray_context):
        """Test basic pub/sub flow with Ray Queue."""
        from dreadnode.core.agents.bus import RayQueueMessageBus

        config = MessageBusConfig(
            producer_topics=["ray.events"],
            consumer_topics=["ray.events"],
        )
        bus = RayQueueMessageBus(config)
        await bus.start()

        # Publish event
        event = SampleEvent(data="ray test message")
        await bus.publish("ray.events", event)

        # Subscribe and receive
        received = []

        async def consume():
            async for evt in bus.subscribe(["ray.events"]):
                received.append(evt)
                break  # Just get one

        # Run consumer with timeout
        try:
            await asyncio.wait_for(consume(), timeout=2.0)
        except asyncio.TimeoutError:
            pass

        assert len(received) == 1
        assert received[0].data == "ray test message"

        await bus.stop()

    @pytest.mark.ray_integration
    @pytest.mark.asyncio
    async def test_multiple_topics(self, ray_context):
        """Test subscribing to multiple topics with Ray Queue."""
        from dreadnode.core.agents.bus import RayQueueMessageBus

        config = MessageBusConfig(
            producer_topics=["ray.topic.a", "ray.topic.b"],
            consumer_topics=["ray.topic.a", "ray.topic.b"],
        )
        bus = RayQueueMessageBus(config)
        await bus.start()

        # Publish to different topics
        await bus.publish("ray.topic.a", SampleEvent(data="from ray A"))
        await bus.publish("ray.topic.b", SampleEvent(data="from ray B"))

        # Subscribe to both
        received = []

        async def consume():
            async for evt in bus.subscribe(["ray.topic.a", "ray.topic.b"]):
                received.append(evt)
                if len(received) >= 2:
                    break

        try:
            await asyncio.wait_for(consume(), timeout=3.0)
        except asyncio.TimeoutError:
            pass

        assert len(received) == 2
        assert {r.data for r in received} == {"from ray A", "from ray B"}

        await bus.stop()

    @pytest.mark.ray_integration
    @pytest.mark.asyncio
    async def test_concurrent_publish(self, ray_context):
        """Test concurrent publishing to Ray Queue."""
        from dreadnode.core.agents.bus import RayQueueMessageBus

        config = MessageBusConfig(
            producer_topics=["ray.concurrent"],
            consumer_topics=["ray.concurrent"],
        )
        bus = RayQueueMessageBus(config)
        await bus.start()

        # Publish multiple events concurrently
        events = [SampleEvent(data=f"msg-{i}") for i in range(10)]
        await asyncio.gather(*[
            bus.publish("ray.concurrent", evt) for evt in events
        ])

        # Consume all
        received = []

        async def consume():
            async for evt in bus.subscribe(["ray.concurrent"]):
                received.append(evt)
                if len(received) >= 10:
                    break

        try:
            await asyncio.wait_for(consume(), timeout=5.0)
        except asyncio.TimeoutError:
            pass

        assert len(received) == 10
        assert {r.data for r in received} == {f"msg-{i}" for i in range(10)}

        await bus.stop()


# =============================================================================
# MessageBus Factory Tests
# =============================================================================


class TestMessageBusFactory:
    """Tests for create_message_bus factory."""

    def test_create_memory_bus(self):
        """Test creating in-memory bus."""
        config = MessageBusConfig()
        bus = create_message_bus("memory", config)
        assert isinstance(bus, InMemoryMessageBus)

    def test_unknown_backend_raises(self):
        """Test that unknown backend raises error."""
        config = MessageBusConfig()
        with pytest.raises(ValueError, match="Unknown backend"):
            create_message_bus("unknown", config)


# =============================================================================
# ActorBase Tests
# =============================================================================


class SimpleTestActor(ActorBase):
    """Simple actor for testing."""

    def __init__(self, config: MessageBusConfig):
        super().__init__(config, backend="memory")
        self.events_received = []
        self.events_published = []

    async def handle_event(self, event: WorkflowEventBase) -> None:
        """Record received events."""
        self.events_received.append(event)

        # Publish a result
        result = ResultEvent(
            task_id=event.task_id,
            result=f"Processed: {getattr(event, 'data', '')}",
        )
        self.events_published.append(result)
        await self.publish(result)


class TestActorBase:
    """Tests for ActorBase."""

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test actor lifecycle."""
        config = MessageBusConfig(
            consumer_topics=["test.events"],
            producer_topics=["test.results"],
        )
        actor = SimpleTestActor(config)

        assert not actor.is_running()

        await actor.start()
        assert actor.is_running()
        assert actor._bus is not None

        await actor.stop()
        assert not actor.is_running()

    @pytest.mark.asyncio
    async def test_publish_requires_start(self):
        """Test that publishing before start raises error."""
        config = MessageBusConfig()
        actor = SimpleTestActor(config)

        event = SampleEvent(data="test")
        with pytest.raises(RuntimeError, match="not started"):
            await actor.publish(event)

    @pytest.mark.asyncio
    async def test_event_processing(self):
        """Test that actor processes events."""
        config = MessageBusConfig(
            consumer_topics=["test.events"],
            producer_topics=["test.results"],
        )
        actor = SimpleTestActor(config)
        await actor.start()

        # Manually call handle_event (simulating bus delivery)
        event = SampleEvent(data="hello world")
        await actor.handle_event(event)

        assert len(actor.events_received) == 1
        assert actor.events_received[0].data == "hello world"
        assert len(actor.events_published) == 1
        assert "Processed: hello world" in actor.events_published[0].result

        await actor.stop()

    @pytest.mark.asyncio
    async def test_consumer_loop(self):
        """Test that consumer loop processes published events."""
        config = MessageBusConfig(
            consumer_topics=["test.events"],
            producer_topics=["test.results"],
        )
        actor = SimpleTestActor(config)
        await actor.start()

        # Publish event to the bus
        event = SampleEvent(data="via bus")
        await actor._bus.publish("test.events", event)

        # Give consumer loop time to process
        await asyncio.sleep(0.3)

        assert len(actor.events_received) == 1
        assert actor.events_received[0].data == "via bus"

        await actor.stop()


# =============================================================================
# Integration Tests
# =============================================================================


class TestBusActorIntegration:
    """Integration tests for bus + actor."""

    @pytest.mark.asyncio
    async def test_two_actors_communicate(self):
        """Test two actors communicating via bus."""
        # Shared bus config
        producer_config = MessageBusConfig(
            producer_topics=["events.out"],
            consumer_topics=[],
        )
        consumer_config = MessageBusConfig(
            producer_topics=[],
            consumer_topics=["events.out"],
        )

        # Create producer (publishes only)
        class ProducerActor(ActorBase):
            async def handle_event(self, event):
                pass

        # Create consumer (receives and records)
        class ConsumerActor(ActorBase):
            def __init__(self, config):
                super().__init__(config, backend="memory")
                self.received = []

            async def handle_event(self, event):
                self.received.append(event)

        producer = ProducerActor(producer_config, backend="memory")
        consumer = ConsumerActor(consumer_config)

        await producer.start()
        await consumer.start()

        # Producer publishes
        event = SampleEvent(data="inter-actor message")
        event.topic = "events.out"
        await producer._bus.publish("events.out", event)

        # Consumer should receive via its bus
        # Note: In-memory buses are separate instances, so this tests the API
        # but not true cross-actor communication (that would need Ray Queue or Kafka)

        await producer.stop()
        await consumer.stop()


# =============================================================================
# ToolExecutor Tests (from multi_turn)
# =============================================================================


class TestToolExecutor:
    """Tests for ToolExecutor from multi_turn module."""

    def test_parse_json_tool_calls(self):
        """Test parsing JSON format tool calls."""
        from dreadnode.core.training.ray.multi_turn import ToolExecutor
        import dreadnode as dn

        @dn.tool
        def calculate(expression: str) -> str:
            """Calculate expression."""
            return str(eval(expression))

        executor = ToolExecutor(tools=[calculate])

        text = 'Let me calculate: {"name": "calculate", "arguments": {"expression": "5 + 3"}}'
        calls = executor.parse_tool_calls(text)

        assert len(calls) == 1
        assert calls[0]["name"] == "calculate"
        assert calls[0]["arguments"]["expression"] == "5 + 3"

    def test_parse_function_tool_calls(self):
        """Test parsing function-style tool calls."""
        from dreadnode.core.training.ray.multi_turn import ToolExecutor
        import dreadnode as dn

        @dn.tool
        def search(query: str) -> str:
            """Search for query."""
            return f"Results for: {query}"

        executor = ToolExecutor(tools=[search])

        text = 'I will search: search(query="python tutorials")'
        calls = executor.parse_tool_calls(text)

        assert len(calls) == 1
        assert calls[0]["name"] == "search"
        assert calls[0]["arguments"]["query"] == "python tutorials"

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test tool execution."""
        from dreadnode.core.training.ray.multi_turn import ToolExecutor
        import dreadnode as dn

        @dn.tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        executor = ToolExecutor(tools=[greet])

        result, success, info = await executor.execute_tool({
            "name": "greet",
            "arguments": {"name": "World"},
        })

        assert success is True
        assert "Hello, World!" in result
        assert info.name == "greet"
        assert info.result is not None

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Test executing unknown tool returns error."""
        from dreadnode.core.training.ray.multi_turn import ToolExecutor

        executor = ToolExecutor(tools=[])

        result, success, info = await executor.execute_tool({
            "name": "nonexistent",
            "arguments": {},
        })

        assert success is False
        assert "Unknown tool" in result
        assert info.error is not None

    def test_get_tool_definitions(self):
        """Test getting tool definitions."""
        from dreadnode.core.training.ray.multi_turn import ToolExecutor
        import dreadnode as dn

        @dn.tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        executor = ToolExecutor(tools=[add])
        defs = executor.get_tool_definitions()

        assert len(defs) == 1
        assert defs[0]["function"]["name"] == "add"

    def test_stats_tracking(self):
        """Test that executor tracks statistics."""
        from dreadnode.core.training.ray.multi_turn import ToolExecutor
        import dreadnode as dn

        @dn.tool
        def echo(text: str) -> str:
            """Echo text."""
            return text

        executor = ToolExecutor(tools=[echo])

        # Execute successfully
        asyncio.get_event_loop().run_until_complete(
            executor.execute_tool({"name": "echo", "arguments": {"text": "hi"}})
        )

        stats = executor.get_stats()
        assert stats["calls_made"] == 1
        assert stats["calls_succeeded"] == 1
        assert stats["calls_failed"] == 0


# =============================================================================
# MultiTurnConfig Tests
# =============================================================================


class TestMultiTurnConfig:
    """Tests for MultiTurnConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from dreadnode.core.training.ray.multi_turn import MultiTurnConfig

        config = MultiTurnConfig()

        assert config.max_turns == 5
        assert config.tools == []
        assert config.stop_on_tool_error is False

    def test_from_grpo_config(self):
        """Test creating from RayGRPOConfig."""
        from dreadnode.core.training.ray.multi_turn import MultiTurnConfig
        from dreadnode.core.training.ray.config import RayGRPOConfig
        import dreadnode as dn

        @dn.tool
        def dummy() -> str:
            """Dummy tool."""
            return "ok"

        grpo_config = RayGRPOConfig(
            model_name="test-model",
            max_new_tokens=256,
            temperature=0.8,
        )

        mt_config = MultiTurnConfig.from_grpo_config(
            grpo_config,
            tools=[dummy],
            max_turns=10,
        )

        assert mt_config.model_name == "test-model"
        assert mt_config.max_new_tokens == 256
        assert mt_config.temperature == 0.8
        assert mt_config.max_turns == 10
        assert len(mt_config.tools) == 1
