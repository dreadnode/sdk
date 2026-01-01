"""
Message bus integration for DN Agents.

Provides:
- MessageBusToolset: Tools for agents to publish events to the message bus
- BusEventHook: Hook that forwards agent events to the bus for observability
"""

import typing as t
from pydantic import Field

from dreadnode.core.tools import Toolset, tool_method
from dreadnode.core.hook import hook
from dreadnode.core.agents.events import (
    AgentEvent,
    AgentStart,
    AgentEnd,
    GenerationStep,
    ToolStart,
    ToolEnd,
    ToolError,
)
from dreadnode.core.agents.reactions import Reaction

from core.bus import MessageBus
from core.events import WorkflowEventBase


class MessageBusToolset(Toolset):
    """
    Toolset providing message bus operations to agents.

    This allows agents to publish events (findings, discoveries, etc.)
    directly to the message bus using natural tool calls.

    Usage:
        bus_tools = MessageBusToolset(
            bus=message_bus,
            source_event=incoming_event,
            event_factory=my_event_factory,
        )

        agent = Agent(
            name="scanner",
            tools=[bus_tools],
            ...
        )
    """

    bus: MessageBus
    """The message bus to publish to."""

    source_event: WorkflowEventBase
    """The event that triggered this agent (for context propagation)."""

    event_factory: t.Callable[[str, dict, WorkflowEventBase], WorkflowEventBase | None] = Field(
        default=lambda t, d, e: None
    )
    """Factory function to create events from (finding_type, data, source_event)."""

    @tool_method
    async def publish_finding(
        self,
        finding_type: str,
        data: dict,
    ) -> str:
        """
        Publish a finding to the message bus.

        Use this tool when you discover something significant that should be
        reported to other actors in the pipeline.

        Args:
            finding_type: Type of finding (e.g., "vulnerability", "idor", "credential", "subdomain")
            data: Finding data as a dictionary with relevant details
        """
        event = self.event_factory(finding_type, data, self.source_event)
        if event is None:
            return f"Unknown finding type: {finding_type}"

        await self.bus.publish(event.topic, event)
        return f"Published {finding_type} finding to {event.topic}"

    @tool_method
    async def publish_discovery(
        self,
        discovery_type: str,
        data: dict,
    ) -> str:
        """
        Publish a discovery to the message bus.

        Use this tool when you discover new assets, endpoints, or data that
        should be processed by other actors.

        Args:
            discovery_type: Type of discovery (e.g., "url", "parameter", "endpoint", "form")
            data: Discovery data as a dictionary
        """
        event = self.event_factory(discovery_type, data, self.source_event)
        if event is None:
            return f"Unknown discovery type: {discovery_type}"

        await self.bus.publish(event.topic, event)
        return f"Published {discovery_type} discovery to {event.topic}"

    @tool_method
    async def request_attack(
        self,
        attack_type: str,
        target: dict,
    ) -> str:
        """
        Request another agent to perform an attack.

        Use this to delegate specific attack patterns to specialized agents.

        Args:
            attack_type: Type of attack to request (e.g., "idor", "sqli", "xss")
            target: Target details including URL, parameters, context
        """
        event = self.event_factory(f"attack_{attack_type}", target, self.source_event)
        if event is None:
            return f"Unknown attack type: {attack_type}"

        await self.bus.publish(event.topic, event)
        return f"Requested {attack_type} attack on {target.get('url', 'unknown')}"


def create_bus_event_hook(
    bus: MessageBus,
    topic_prefix: str = "agent.events",
) -> t.Callable[[AgentEvent], t.Awaitable[Reaction | None]]:
    """
    Create a hook that forwards agent events to the message bus.

    This enables observability and allows other systems to react to
    agent execution in real-time.

    Args:
        bus: The message bus to publish to
        topic_prefix: Prefix for event topics (e.g., "agent.events.generation")

    Returns:
        A hook function compatible with DN Agent.hooks
    """

    async def bus_event_hook(event: AgentEvent) -> Reaction | None:
        """Forward agent events to the message bus."""
        event_type = type(event).__name__.lower()
        topic = f"{topic_prefix}.{event_type}"

        # Convert to a serializable format
        event_data = {
            "agent_id": str(event.agent_id),
            "agent_name": event.agent_name,
            "status": event.status,
            "timestamp": event.timestamp.isoformat(),
            "event_type": event_type,
        }

        # Add event-specific data
        if isinstance(event, AgentStart):
            event_data["inputs"] = event.inputs
            event_data["params"] = event.params
        elif isinstance(event, AgentEnd):
            event_data["stop_reason"] = event.stop_reason
            if event.error:
                event_data["error"] = str(event.error)
        elif isinstance(event, GenerationStep):
            event_data["step"] = event.step
            event_data["usage"] = {
                "input_tokens": event.usage.input_tokens,
                "output_tokens": event.usage.output_tokens,
                "total_tokens": event.usage.total_tokens,
            }
        elif isinstance(event, (ToolStart, ToolEnd)):
            event_data["tool_name"] = event.tool_call.name
            event_data["tool_call_id"] = event.tool_call.id
            if isinstance(event, ToolEnd) and event.result:
                event_data["result"] = event.result[:500]  # Truncate
        elif isinstance(event, ToolError):
            event_data["tool_name"] = event.tool_call.name
            event_data["error"] = str(event.error)

        # Publish (fire and forget - don't block agent execution)
        try:
            await bus.publish(topic, event_data)
        except Exception:
            pass  # Don't let bus errors affect agent

        return None  # Never modify agent behavior

    return bus_event_hook


@hook(GenerationStep)
async def log_generation_hook(event: GenerationStep) -> Reaction | None:
    """Simple hook that logs generation steps."""
    print(
        f"[Agent:{event.agent_name}] Step {event.step}: "
        f"{event.usage.total_tokens} tokens"
    )
    return None


@hook(ToolEnd)
async def log_tool_hook(event: ToolEnd) -> Reaction | None:
    """Simple hook that logs tool completions."""
    result_preview = (event.result or "")[:100]
    print(f"[Agent:{event.agent_name}] Tool {event.tool_call.name}: {result_preview}")
    return None
