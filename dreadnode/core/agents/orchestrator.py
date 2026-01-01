import asyncio
from typing import TYPE_CHECKING

from dreadnode.core.agents.bus import MessageBus, MessageBusConfig, create_message_bus

if TYPE_CHECKING:
    from dreadnode.core.agents.events import WorkflowEventBase


class ActorBase:
    """
    Base class for Ray Actor event processors.

    This replaces the old OrchestratorBase that was designed for Ray Serve.
    It's designed for @ray.remote actors that:
    - Consume events from a message bus
    - Process events with streaming (results published as they arrive)
    - Produce events back to the message bus

    Subclasses should implement:
    - handle_event(event): Process a single event

    Example:
        @ray.remote
        class MyTool(ActorBase):
            async def handle_event(self, event):
                async for result in process(event):
                    await self.publish(ResultEvent(...))

        # Start the actor
        actor = MyTool.remote(config)
        await actor.start.remote()
    """

    def __init__(
        self,
        config: MessageBusConfig,
        event_parser=None,
        backend: str = "kafka",
    ):
        """
        Initialize the actor base.

        Args:
            config: MessageBusConfig with bus settings
            event_parser: Pydantic TypeAdapter for event validation
            backend: Message bus backend ("kafka", "ray", "memory")
        """
        self.config = config
        self.event_parser = event_parser
        self.backend = backend
        self._bus: MessageBus | None = None
        self._running = False
        self._consumer_task: asyncio.Task | None = None

    async def start(self) -> None:
        """
        Initialize the message bus and start consuming events.

        This should be called after the actor is created:
            actor = MyActor.remote(config)
            await actor.start.remote()
        """
        print(f"[{self.__class__.__name__}] Starting actor...")

        # Create and start the message bus
        self._bus = create_message_bus(
            backend=self.backend,
            config=self.config,
            event_parser=self.event_parser,
        )
        await self._bus.start()

        self._running = True

        # Start the consumer loop if we have topics to consume
        if self.config.consumer_topics:
            print(
                f"[{self.__class__.__name__}] Subscribing to topics: {self.config.consumer_topics}"
            )
            self._consumer_task = asyncio.create_task(self._consume_loop())

        print(f"[{self.__class__.__name__}] Actor started successfully")

    async def stop(self) -> None:
        """Gracefully shut down the actor."""
        print(f"[{self.__class__.__name__}] Stopping actor...")

        self._running = False

        # Cancel consumer task
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

        # Stop the message bus
        if self._bus:
            await self._bus.stop()

        print(f"[{self.__class__.__name__}] Actor stopped")

    async def _consume_loop(self) -> None:
        """Main loop that consumes events and dispatches to handle_event."""
        try:
            async for event in self._bus.subscribe():
                if not self._running:
                    break

                try:
                    print(
                        f"[{self.__class__.__name__}] Received event: {event.topic}",
                        flush=True,
                    )
                    await self.handle_event(event)

                except Exception as e:
                    print(f"[{self.__class__.__name__}] Error processing event: {e}")
                    # Continue processing other events

        except asyncio.CancelledError:
            print(f"[{self.__class__.__name__}] Consumer loop cancelled")

    async def publish(self, event: "WorkflowEventBase") -> None:
        """
        Publish an event to the message bus.

        The topic is determined by the event's `topic` field.

        Args:
            event: The event to publish
        """
        if not self._bus:
            raise RuntimeError("Actor not started. Call start() first.")

        topic = event.topic
        await self._bus.publish(topic, event)
        print(f"[{self.__class__.__name__}] Published: {event.topic}")

    async def publish_and_wait(self, event: "WorkflowEventBase") -> None:
        """Publish an event and wait for acknowledgment."""
        if not self._bus:
            raise RuntimeError("Actor not started. Call start() first.")

        topic = event.topic
        await self._bus.publish_and_wait(topic, event)
        print(f"[{self.__class__.__name__}] Published (confirmed): {event.topic}")

    async def handle_event(self, event: "WorkflowEventBase") -> None:
        """
        Process a single event.

        Subclasses must implement this method to define their processing logic.
        For streaming results, publish events as they become available:

            async def handle_event(self, event):
                async for result in self._run_tool(event):
                    await self.publish(ResultEvent(data=result, **event.context()))

        Args:
            event: The event to process
        """
        raise NotImplementedError("Subclasses must implement handle_event()")

    def is_running(self) -> bool:
        """Check if the actor is running."""
        return self._running
