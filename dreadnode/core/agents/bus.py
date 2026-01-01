"""
Message bus abstraction for swappable pub/sub backends.

Supported backends:
- Kafka (production)
- Ray Queue (simpler, Ray-native)
- In-memory (testing)

Usage:
    bus = KafkaMessageBus(config)
    await bus.start()

    # Publishing
    await bus.publish("topic.name", event)

    # Subscribing (streaming)
    async for event in bus.subscribe(["topic.a", "topic.b"]):
        await process(event)
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from dreadnode.core.agents.events import WorkflowEventBase


@dataclass
class MessageBusConfig:
    """Configuration for message bus backends."""

    # Kafka settings
    bootstrap_servers: str = "localhost:9092"

    # Consumer settings
    group_id: str | None = None
    consumer_topics: list[str] = field(default_factory=list)
    auto_offset_reset: str = "earliest"

    # Producer settings
    producer_topics: list[str] = field(default_factory=list)

    # Ray Queue settings (for RayQueueMessageBus)
    queue_max_size: int = 10000


class MessageBus(ABC):
    """
    Abstract message bus interface.

    Implementations must support:
    - Async start/stop lifecycle
    - Publishing events to topics
    - Subscribing to topics with streaming iteration
    """

    @abstractmethod
    async def start(self) -> None:
        """Initialize connections and start the bus."""

    @abstractmethod
    async def stop(self) -> None:
        """Gracefully shut down the bus."""

    @abstractmethod
    async def publish(self, topic: str, event: WorkflowEventBase) -> None:
        """Publish an event to a topic."""

    @abstractmethod
    async def publish_and_wait(self, topic: str, event: WorkflowEventBase) -> None:
        """Publish an event and wait for acknowledgment."""

    @abstractmethod
    def subscribe(self, topics: list[str]) -> AsyncIterator[WorkflowEventBase]:
        """
        Subscribe to topics and yield events as they arrive.

        This is a streaming interface - it yields events continuously
        until the bus is stopped.
        """


class KafkaMessageBus(MessageBus):
    """Kafka-based message bus implementation."""

    def __init__(self, config: MessageBusConfig, event_parser=None):
        self.config: MessageBusConfig = config
        self.event_parser = event_parser
        self._producer = None
        self._consumer = None
        self._running = False

    async def start(self) -> None:
        from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

        # Initialize producer if we have topics to publish to
        if self.config.producer_topics:
            self._producer = AIOKafkaProducer(bootstrap_servers=self.config.bootstrap_servers)
            await self._producer.start()

        # Initialize consumer if we have topics to subscribe to
        if self.config.consumer_topics and self.config.group_id:
            self._consumer = AIOKafkaConsumer(
                *self.config.consumer_topics,
                bootstrap_servers=self.config.bootstrap_servers,
                group_id=self.config.group_id,
                auto_offset_reset=self.config.auto_offset_reset,
            )
            await self._consumer.start()

        self._running = True

    async def stop(self) -> None:
        self._running = False

        if self._consumer:
            await self._consumer.stop()
            self._consumer = None

        if self._producer:
            await self._producer.stop()
            self._producer = None

    async def publish(self, topic: str, event: WorkflowEventBase) -> None:
        if not self._producer:
            raise RuntimeError("Producer not initialized. Call start() first.")

        await self._producer.send(
            topic,
            value=event.model_dump_json().encode("utf-8"),
            key=event.task_id.encode("utf-8") if event.task_id else None,
        )

    async def publish_and_wait(self, topic: str, event: WorkflowEventBase) -> None:
        if not self._producer:
            raise RuntimeError("Producer not initialized. Call start() first.")

        await self._producer.send_and_wait(
            topic,
            value=event.model_dump_json().encode("utf-8"),
            key=event.task_id.encode("utf-8") if event.task_id else None,
        )

    async def subscribe(self, topics: list[str] = None) -> AsyncIterator[WorkflowEventBase]:
        """Yield events from subscribed topics."""
        if not self._consumer:
            raise RuntimeError("Consumer not initialized. Call start() first.")

        async for msg in self._consumer:
            if not self._running:
                break

            try:
                if self.event_parser:
                    event = self.event_parser.validate_json(msg.value)
                    yield event
                else:
                    yield msg.value
            except Exception as e:
                print(f"[KafkaMessageBus] Error parsing message: {e}")
                continue


class RayQueueMessageBus(MessageBus):
    """
    Ray-native message bus using ray.util.Queue.

    Simpler than Kafka, good for:
    - Development/testing
    - Single-cluster deployments
    - Lower latency requirements

    Limitations:
    - No persistence (messages lost on restart)
    - No replay capability
    - Single cluster only
    """

    def __init__(self, config: MessageBusConfig, event_parser=None):
        from ray.util.queue import Queue

        self.config = config
        self.event_parser = event_parser
        self._queues: dict[str, Queue] = {}
        self._running = False

    async def start(self) -> None:
        from ray.util.queue import Queue

        # Create queues for all topics we'll use
        all_topics = set(self.config.producer_topics + self.config.consumer_topics)
        for topic in all_topics:
            if topic not in self._queues:
                self._queues[topic] = Queue(maxsize=self.config.queue_max_size)

        self._running = True

    async def stop(self) -> None:
        self._running = False
        # Queues are managed by Ray, no explicit cleanup needed

    async def publish(self, topic: str, event: WorkflowEventBase) -> None:
        from ray.util.queue import Queue

        if topic not in self._queues:
            self._queues[topic] = Queue(maxsize=self.config.queue_max_size)

        await self._queues[topic].put_async(event)

    async def publish_and_wait(self, topic: str, event: WorkflowEventBase) -> None:
        # For Ray Queue, put is already synchronous in effect
        await self.publish(topic, event)

    async def subscribe(self, topics: list[str] = None) -> AsyncIterator[WorkflowEventBase]:
        """Yield events from subscribed queues."""
        import asyncio

        topics = topics or self.config.consumer_topics

        # Ensure queues exist
        for topic in topics:
            if topic not in self._queues:
                from ray.util.queue import Queue

                self._queues[topic] = Queue(maxsize=self.config.queue_max_size)

        while self._running:
            # Poll all subscribed queues
            for topic in topics:
                try:
                    event = await asyncio.wait_for(
                        self._queues[topic].get_async(),
                        timeout=0.1,  # Short timeout to check other queues
                    )
                    yield event
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"[RayQueueMessageBus] Error: {e}")
                    continue


class InMemoryMessageBus(MessageBus):
    """
    Simple in-memory message bus for testing.

    Uses asyncio.Queue for each topic.
    """

    def __init__(self, config: MessageBusConfig = None, event_parser=None):
        import asyncio

        self.config = config or MessageBusConfig()
        self.event_parser = event_parser
        self._queues: dict[str, asyncio.Queue] = {}
        self._running = False

    async def start(self) -> None:
        import asyncio

        all_topics = set((self.config.producer_topics or []) + (self.config.consumer_topics or []))
        for topic in all_topics:
            self._queues[topic] = asyncio.Queue()
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def publish(self, topic: str, event: WorkflowEventBase) -> None:
        import asyncio

        if topic not in self._queues:
            self._queues[topic] = asyncio.Queue()
        await self._queues[topic].put(event)

    async def publish_and_wait(self, topic: str, event: WorkflowEventBase) -> None:
        await self.publish(topic, event)

    async def subscribe(self, topics: list[str] = None) -> AsyncIterator[WorkflowEventBase]:
        import asyncio

        topics = topics or self.config.consumer_topics or []

        for topic in topics:
            if topic not in self._queues:
                self._queues[topic] = asyncio.Queue()

        while self._running:
            for topic in topics:
                try:
                    event = await asyncio.wait_for(self._queues[topic].get(), timeout=0.1)
                    yield event
                except asyncio.TimeoutError:
                    continue


def create_message_bus(backend: str, config: MessageBusConfig, event_parser=None) -> MessageBus:
    """
    Factory function to create a message bus.

    Args:
        backend: One of "kafka", "ray", "memory"
        config: MessageBusConfig instance
        event_parser: Optional Pydantic TypeAdapter for event parsing

    Returns:
        MessageBus implementation
    """
    backends = {
        "kafka": KafkaMessageBus,
        "ray": RayQueueMessageBus,
        "memory": InMemoryMessageBus,
    }

    if backend not in backends:
        raise ValueError(f"Unknown backend: {backend}. Choose from: {list(backends.keys())}")

    return backends[backend](config, event_parser)
