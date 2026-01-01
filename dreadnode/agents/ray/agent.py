"""
Base class for LLM-powered Ray Actor Agents.

Integrates DN Agent SDK with Ray Actors and MessageBus.

Trajectory Export:
    DN SDK provides built-in trajectory export via `dn.run()` context.
    For distributed Ray actors, we use TrajectoryEvent to publish to
    the message bus, using DN's trajectory utilities for serialization.
"""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any

from dreadnode.agents.ray.threaded import AgentHandoff, HandoffToolset
from dreadnode.agents.ray.tools import MessageBusToolset
from dreadnode.core.agents import Agent
from dreadnode.core.agents.bus import MessageBusConfig
from dreadnode.core.agents.events import (
    AgentEnd,
    AgentEvent,
    GenerationStep,
    ToolError,
    ToolStart,
    WorkflowEventBase,
)
from dreadnode.core.agents.orchestrator import ActorBase
from dreadnode.core.agents.trajectory import (
    Trajectory,
    trajectory_to_jsonl_record,
)
from dreadnode.core.tools import Tool, Toolset

EventFactory = Callable[[str, dict, WorkflowEventBase], WorkflowEventBase | None]


class AgentActorBase(ActorBase):
    """
    Base class for Ray Actors with LLM Agents.

    Features:
    - Converts bus events to agent goals
    - Publishes agent output as bus events
    - Supports agent handoffs with trajectory passing

    Example:
        @ray.remote
        class MyAgent(AgentActorBase):
            def __init__(self, config):
                super().__init__(
                    config=config,
                    agent_name="My Agent",
                    agent_instructions="You are a helpful agent...",
                    tools=[MyToolset()],
                )

            async def event_to_goal(self, event) -> str | None:
                if isinstance(event, MyEvent):
                    return f"Process: {event.data}"
                return None
    """

    def __init__(
        self,
        config: MessageBusConfig,
        event_parser=None,
        backend: str = "kafka",
        agent_name: str = "agent",
        agent_model: str = "anthropic/claude-sonnet-4-20250514",
        agent_instructions: str = "",
        tools: list[Tool | Toolset] | None = None,
        max_steps: int = 25,
        hooks: list | None = None,
        event_factory: EventFactory | None = None,
        *,
        publish_trajectories: bool = False,
        task_type: str = "",
    ):
        super().__init__(config, event_parser, backend)

        self.agent_name = agent_name
        self.agent_model = agent_model
        self.agent_instructions = agent_instructions
        self.tools = tools or []
        self.max_steps = max_steps
        self.hooks = hooks or []
        self.event_factory = event_factory or (lambda t, d, s: None)

        # Trajectory collection for training
        self.publish_trajectories = publish_trajectories
        self.task_type = task_type

    @abstractmethod
    async def event_to_goal(self, event: WorkflowEventBase) -> str | None:
        """Convert bus event to agent goal. Return None to skip."""

    async def handle_event(self, event: WorkflowEventBase) -> None:
        """Process event with agent."""
        goal = await self.event_to_goal(event)
        if goal is None:
            return

        # Build tools list
        tools = list(self.tools)

        # Add bus publishing tools
        tools.append(
            MessageBusToolset(
                bus=self._bus,
                source_event=event,
                event_factory=self.event_factory,
            )
        )

        # Create agent
        agent = Agent(
            name=self.agent_name,
            model=self.agent_model,
            instructions=self.agent_instructions,
            tools=tools,
            max_steps=self.max_steps,
            hooks=list(self.hooks),
        )

        # For handoffs, inject prior messages
        if isinstance(event, AgentHandoff):
            agent.trajectory.messages = event.to_messages()

        # Add handoff tools with reference to agent's trajectory
        handoff_tools = HandoffToolset(
            bus=self._bus,
            agent_name=self.agent_name,
            trajectory=agent.trajectory,  # Reference - will update during execution
            source_event=event,
        )
        agent.tools.append(handoff_tools)

        try:
            async with agent.stream(goal) as stream:
                async for agent_event in stream:
                    self._log_agent_event(agent_event, event.task_id)

            await self.on_agent_complete(agent.trajectory, event)

        except Exception as e:
            print(f"[{self.agent_name}] Error: {e}")
            await self.on_agent_error(e, event)

    def _log_agent_event(self, agent_event: AgentEvent, task_id: str) -> None:
        prefix = f"[{self.agent_name}][{task_id}]"
        match agent_event:
            case GenerationStep():
                print(f"{prefix} Step {agent_event.step}")
            case ToolStart():
                print(f"{prefix} Tool: {agent_event.tool_call.name}")
            case ToolError():
                print(f"{prefix} Tool error: {agent_event.error}")
            case AgentEnd():
                print(f"{prefix} Done: {agent_event.stop_reason}")

    async def on_agent_complete(
        self, trajectory: Trajectory, source_event: WorkflowEventBase
    ) -> None:
        """Called when agent completes. Override to publish results."""
        # Optionally publish trajectory for training
        if self.publish_trajectories:
            await self._publish_trajectory(trajectory, source_event, success=True)

    async def on_agent_error(self, error: Exception, source_event: WorkflowEventBase) -> None:
        """Called when agent errors. Override to handle errors."""

    async def _publish_trajectory(
        self,
        trajectory: Trajectory,
        source_event: WorkflowEventBase,
        success: bool | None = None,
        reward: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Publish trajectory for training data collection.

        Uses DN SDK's trajectory_to_jsonl_record() for consistent format
        with DN's built-in tracing output.

        Args:
            trajectory: The agent's trajectory
            source_event: The event that triggered this agent
            success: Whether the task was successful
            reward: Optional reward signal
            metadata: Additional metadata
        """
        from services.training.core.events import TrajectoryEvent

        # Extract tool schemas from tools
        tool_schemas = []
        for tool in self.tools:
            if isinstance(tool, Toolset):
                tool_schemas.extend([t.api_definition for t in tool.get_tools()])
            elif isinstance(tool, Tool):
                tool_schemas.append(tool.api_definition)

        record = trajectory_to_jsonl_record(
            trajectory,
            system_prompt=self.agent_instructions,
            tools=tool_schemas if tool_schemas else None,
            metadata={
                "agent_name": self.agent_name,
                "task_type": self.task_type or source_event.topic,
                "job_id": source_event.job_id,
                "correlation_id": source_event.correlation_id,
                "success": success,
                "reward": reward,
                "total_steps": len(trajectory.steps),
                "total_tokens": trajectory.usage.total_tokens,
                **(metadata or {}),
            },
        )

        event = TrajectoryEvent(
            job_id=source_event.job_id,
            correlation_id=source_event.correlation_id,
            agent_name=self.agent_name,
            task_type=self.task_type or source_event.topic,
            messages=record["messages"],
            tools=record.get("tools", []),
            success=success,
            reward=reward,
            metadata=record.get("metadata", {}),
        )

        await self._bus.publish(event)
        print(f"[{self.agent_name}] Published trajectory ({len(record['messages'])} messages)")
