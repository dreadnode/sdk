"""
Threaded agent communication using DN Trajectory.

Events carry serialized trajectory, enabling agents to
continue each other's conversations.
"""

from pydantic import Field

from dreadnode.core.tools import Toolset, tool_method
from dreadnode.core.agents.trajectory import Trajectory
from dreadnode.core.generators.message import Message

from core.bus import MessageBus
from core.events import WorkflowEventBase


class AgentHandoff(WorkflowEventBase):
    """
    Event that passes conversation context between agents.

    Carries the trajectory so the receiving agent can continue
    the conversation with full history.
    """
    topic: str = "agent.handoff"

    from_agent: str = Field(...)
    to_agent: str = Field(...)

    # The request/question for the receiving agent
    request: str = Field(...)

    # Serialized trajectory - the conversation so far
    trajectory_messages: list[dict] = Field(default_factory=list)

    # Optional structured data
    data: dict = Field(default_factory=dict)

    @classmethod
    def from_trajectory(
        cls,
        trajectory: Trajectory,
        from_agent: str,
        to_agent: str,
        request: str,
        source_event: WorkflowEventBase,
        data: dict | None = None,
    ) -> "AgentHandoff":
        """Create handoff from a trajectory."""
        # Serialize messages
        messages = [
            {"role": m.role, "content": m.content}
            for m in trajectory.messages
        ]

        return cls(
            topic=f"agent.handoff.{to_agent}",
            from_agent=from_agent,
            to_agent=to_agent,
            request=request,
            trajectory_messages=messages,
            data=data or {},
            task_id=source_event.task_id,
            **source_event.context(),
        )

    def to_messages(self) -> list[Message]:
        """Convert back to Message objects."""
        return [
            Message(role=m["role"], content=m["content"])
            for m in self.trajectory_messages
        ]


class HandoffToolset(Toolset):
    """
    Tools for agents to hand off conversations to other agents.
    """

    bus: MessageBus
    agent_name: str
    trajectory: Trajectory | None = None
    source_event: WorkflowEventBase | None = None

    @tool_method
    async def handoff_to(
        self,
        to_agent: str,
        request: str,
        data: dict | None = None,
    ) -> str:
        """
        Hand off the conversation to another agent.

        The receiving agent will see the full conversation history
        and can continue from where you left off.

        Args:
            to_agent: Name of the agent to hand off to
            request: What you want them to do
            data: Optional structured data to include
        """
        if not self.trajectory:
            return "Error: No trajectory to hand off"

        handoff = AgentHandoff.from_trajectory(
            trajectory=self.trajectory,
            from_agent=self.agent_name,
            to_agent=to_agent,
            request=request,
            source_event=self.source_event,
            data=data,
        )

        await self.bus.publish(handoff.topic, handoff)

        return f"Handed off to {to_agent} with full conversation context"

    @tool_method
    async def reply_to_handoff(
        self,
        response: str,
        data: dict | None = None,
    ) -> str:
        """
        Reply to the agent who handed off to you.

        Args:
            response: Your response/findings
            data: Optional structured data
        """
        if not isinstance(self.source_event, AgentHandoff):
            return "Error: Not responding to a handoff"

        if not self.trajectory:
            return "Error: No trajectory to hand off"

        original = self.source_event

        handoff = AgentHandoff.from_trajectory(
            trajectory=self.trajectory,
            from_agent=self.agent_name,
            to_agent=original.from_agent,
            request=response,
            source_event=original,
            data=data,
        )

        await self.bus.publish(handoff.topic, handoff)

        return f"Replied to {original.from_agent}"
