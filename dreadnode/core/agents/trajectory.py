import typing as t

from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID

from dreadnode.core.agents.events import AgentEnd, AgentEvent, AgentStep, GenerationStep, ToolStep
from dreadnode.core.generators.generator import Usage
from dreadnode.core.generators.message import Message
from dreadnode.core.metric import MetricSeries

AgentEventT = t.TypeVar("AgentEventT", bound=AgentEvent)


class Trajectory(BaseModel):
    """
    The Trajectory creates ordered sequence of all events and steps for a single agent run.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: ULID = Field(default_factory=ULID)
    """The unique identifier for this agent session."""
    agent_id: ULID | None = None
    """The unique identifier for the agent associated with this trajectory."""
    events: list[AgentEvent] = Field(default_factory=list)
    """The ordered list of events and steps in this trajectory."""
    scores: dict[str, MetricSeries] = Field(default_factory=dict)
    """Scores accumulated during agent execution via ScorerHooks, keyed by scorer name."""

    @property
    def steps(self) -> list[AgentStep]:
        """Returns only the AgentStep instances from the event history."""
        return [event for event in self.events if isinstance(event, AgentStep)]

    @property
    def messages(self) -> list[Message]:
        """Returns the reconstructed history from all steps."""
        msgs = []
        for step in self.steps:
            # Since each step only contains the delta, we can safely extend
            msgs.extend(step.messages)
        return msgs

    @property
    def usage(self) -> Usage:
        """Calculates the total usage from all steps in the trajectory."""
        total = Usage(input_tokens=0, output_tokens=0, total_tokens=0)
        for step in self.steps:
            total += step.usage
        return total

    def add_event(self, event: AgentEvent) -> None:
        """Adds a new event or step to the trajectory."""
        self.events.append(event)

    def log_score(
        self,
        scorer_name: str,
        value: float,
        step: int | None = None,
    ) -> None:
        """
        Log a score from a ScorerHook execution.

        Args:
            scorer_name: The name of the scorer.
            value: The score value.
            step: The step number (defaults to current step count).
        """
        if scorer_name not in self.scores:
            self.scores[scorer_name] = MetricSeries()
        self.scores[scorer_name].append(value, step=step or len(self.steps))

    # No changes needed for get_events_by_type, but it's more intuitive now
    def get_events_by_type(self, event_type: type[AgentEventT]) -> list[AgentEventT]:
        return [event for event in self.events if isinstance(event, event_type)]

    def get_summary(self) -> str:
        agent_end = self.get_events_by_type(AgentEnd)[-1]

        return (
            f"Agent '{self.agent_id}' finished: "
            f"reason='{agent_end.stop_reason}', "
            f"steps={len(self.steps)}, "
            f"total_tokens={self.usage.total_tokens}, "
            f"in_tokens={self.usage.input_tokens}, "
            f"out_tokens={self.usage.output_tokens}"
        )

    # In Trajectory:

    def _format_single_step(self, step: AgentStep) -> list[dict[str, t.Any]]:
        if not step.messages:
            return []

        rows = []

        base_payload = {
            "session_id": str(self.session_id),
            "step": step.step,
            "timestamp": step.timestamp.isoformat(),
            "tokens": step.usage.total_tokens if step.usage else 0,
            "type": None,
            "role": None,
            "content": None,
            "tool_calls": None,
            "tool_call_id": None,
            "name": None,
        }

        if step.step == 1:
            context_messages = step.messages[:-1]

            for msg in context_messages:
                row = base_payload.copy()
                row.update(
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "type": "context",
                    }
                )
                rows.append(row)

        # --- CURRENT STEP LOGIC ---
        last_msg = step.messages[-1]
        current_row = base_payload.copy()

        if isinstance(step, GenerationStep):
            current_row.update(
                {
                    "type": "generation",
                    "role": "assistant",
                    "content": last_msg.content,
                    "tool_calls": [t.model_dump() for t in last_msg.tool_calls]
                    if last_msg.tool_calls
                    else None,
                }
            )
            rows.append(current_row)

        elif isinstance(step, ToolStep):
            current_row.update(
                {
                    "type": "tool_call",
                    "role": "tool",
                    "content": last_msg.content,
                    "tool": getattr(last_msg, "name", step.tool_call.name),
                }
            )
            rows.append(current_row)

        return rows

    def log_step(self, step: AgentStep) -> None:
        from dreadnode import log_outputs

        self.events.append(step)

        payload = self._format_single_step(step)

        if payload:
            for row in payload:
                log_outputs(**row)

    def trajectory_to_rows(self) -> list[dict[str, t.Any]]:
        rows = []

        session_meta = {
            "session_id": str(self.session_id),
            "agent_id": self.agent_id,
        }

        global_sequence_id = 0

        for step in self.steps:
            if not hasattr(step, "messages") or not step.messages:
                continue

            usage_meta = {}
            if isinstance(step, AgentStep) and step.usage:
                usage_meta = {
                    "input_tokens": step.usage.input_tokens,
                    "output_tokens": step.usage.output_tokens,
                    "total_tokens": step.usage.total_tokens,
                }

            for msg in step.messages:
                row = {
                    **session_meta,
                    "sequence_id": global_sequence_id,
                    "step_index": step.step if hasattr(step, "step") else 0,
                    "role": msg.role,
                    "content": msg.content,
                    "has_tool_calls": bool(msg.tool_calls),
                    "tool_calls_json": msg.tool_calls if msg.tool_calls else None,
                    **usage_meta,
                }
                rows.append(row)
                global_sequence_id += 1

        return rows


def trajectory_to_turns(trajectory: Trajectory) -> list[dict[str, t.Any]]:
    rows = []

    final_messages = trajectory.messages
    system_prompt = next((m.content for m in final_messages if m.role == "system"), None)

    for event in trajectory.steps:
        if not isinstance(event, AgentStep) or not event.messages:
            continue

        if event.messages[-1].role != "assistant":
            continue

        history = event.messages
        assistant_msg = history[-1]

        user_content = None
        if len(history) >= 2:
            prev_msg = history[-2]
            user_content = prev_msg.content

            if prev_msg.role == "tool":
                user_content = f"[Tool Result from {prev_msg.tool_call_id}]\n{user_content}"

        rows.append(
            {
                "session_id": str(trajectory.session_id),
                "step": event.step,
                "system": system_prompt,
                "user": user_content,
                "assistant": assistant_msg.content,
                "tool_calls": [t.model_dump() for t in assistant_msg.tool_calls]
                if assistant_msg.tool_calls
                else None,
                "usage_tokens": event.usage.total_tokens if event.usage else 0,
            }
        )

    return rows


def trajectory_to_openai_format(trajectory: Trajectory) -> list[dict[str, t.Any]]:
    """
    Convert a DN Agent Trajectory to OpenAI-compatible message format.

    This format is compatible with NeMo RL's OpenAIFormatDataset.

    Args:
        trajectory: DN Agent Trajectory object

    Returns:
        List of OpenAI-format messages with role, content, tool_calls, tool_call_id
    """
    messages = []

    for msg in trajectory.messages:
        message: dict[str, t.Any] = {
            "role": msg.role,
        }

        # Handle content - could be string or structured
        if msg.content is not None:
            message["content"] = msg.content
        else:
            message["content"] = ""

        # Handle tool calls (assistant messages that invoke tools)
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments
                        if isinstance(tc.arguments, str)
                        else str(tc.arguments),
                    },
                }
                for tc in msg.tool_calls
            ]

        # Handle tool call ID (tool result messages)
        if hasattr(msg, "tool_call_id") and msg.tool_call_id:
            message["tool_call_id"] = msg.tool_call_id

        messages.append(message)

    return messages


def trajectory_from_openai_format(
    messages: list[dict[str, t.Any]],
    message_class: type | None = None,
) -> "Trajectory":
    """
    Create a Trajectory from OpenAI-format messages.

    Args:
        messages: List of OpenAI-format message dicts
        message_class: Optional Message class to use (defaults to importing from dreadnode)

    Returns:
        Trajectory instance

    Example:
        >>> trajectory = trajectory_from_openai_format([
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there!"}
        ... ])
    """
    from dreadnode.core.agents.trajectory import Trajectory
    from dreadnode.core.generators.message import Message

    if message_class is None:
        message_class = Message

    trajectory = Trajectory()

    for msg in messages:
        message = message_class(
            role=msg.get("role", "user"),
            content=msg.get("content", ""),
        )
        # Note: tool_calls and tool_call_id would need Message class support
        trajectory.messages.append(message)

    return trajectory


def trajectory_to_jsonl_record(
    trajectory: "Trajectory",
    system_prompt: str | None = None,
    tools: list[dict] | None = None,
    metadata: dict[str, t.Any] | None = None,
) -> dict[str, t.Any]:
    """
    Convert trajectory to a JSONL record for training data export.

    This produces a record compatible with NeMo RL, OpenAI fine-tuning,
    and other frameworks that accept OpenAI-format training data.

    Args:
        trajectory: The trajectory to convert
        system_prompt: Optional system prompt to prepend
        tools: Optional tool definitions used by the agent
        metadata: Optional metadata to include (agent_name, task_type, etc.)

    Returns:
        Dict ready for JSON serialization

    Example:
        >>> record = trajectory_to_jsonl_record(
        ...     agent.trajectory,
        ...     system_prompt="You are a helpful assistant.",
        ...     metadata={"agent_name": "MyAgent", "success": True}
        ... )
        >>> with open("training.jsonl", "a") as f:
        ...     f.write(json.dumps(record) + "\\n")
    """
    messages = trajectory_to_openai_format(trajectory)

    # Prepend system prompt if provided and not already present
    if system_prompt and (not messages or messages[0].get("role") != "system"):
        messages.insert(0, {"role": "system", "content": system_prompt})

    record: dict[str, t.Any] = {"messages": messages}

    if tools:
        record["tools"] = tools

    if metadata:
        record["metadata"] = metadata

    return record


def trajectories_to_hf_dataset(
    trajectories: list[dict[str, t.Any]],
    format: str = "messages",
) -> "Dataset":
    """
    Convert trajectories to a Hugging Face Dataset.

    Args:
        trajectories: List of trajectory dicts
        format: Output format - "messages" (OpenAI), "chat" (TRL), or "turns"

    Returns:
        HF Dataset ready for training

    Example:
        >>> from services.training import load_trajectory_jsonl, trajectories_to_hf_dataset
        >>> trajectories = load_trajectory_jsonl("./training.jsonl")
        >>> dataset = trajectories_to_hf_dataset(trajectories, format="chat")
        >>> dataset.push_to_hub("my-org/agent-trajectories")
    """
    from datasets import Dataset

    if format == "messages":
        # OpenAI format - list of messages per row
        return Dataset.from_list(
            [
                {
                    "messages": traj["messages"],
                    "tools": traj.get("tools", []),
                    **traj.get("metadata", {}),
                }
                for traj in trajectories
            ]
        )

    if format == "chat":
        # TRL chat format - conversation as structured field
        rows = []
        for traj in trajectories:
            messages = traj["messages"]
            metadata = traj.get("metadata", {})

            # Extract system, user, assistant for TRL SFTTrainer
            conversation = []
            for msg in messages:
                conversation.append(
                    {
                        "role": msg["role"],
                        "content": msg.get("content", ""),
                    }
                )

            rows.append(
                {
                    "conversation": conversation,
                    "agent_name": metadata.get("agent_name", "unknown"),
                    "task_type": metadata.get("task_type", "unknown"),
                    "success": metadata.get("success"),
                }
            )

        return Dataset.from_list(rows)

    if format == "turns":
        # Flattened turn format - one row per assistant turn
        rows = []
        for traj in trajectories:
            messages = traj["messages"]
            metadata = traj.get("metadata", {})

            system_prompt = None
            history = []

            for msg in messages:
                role = msg["role"]
                content = msg.get("content", "")

                if role == "system":
                    system_prompt = content
                elif role == "user":
                    history.append({"role": "user", "content": content})
                elif role == "assistant":
                    # Create a row for each assistant response
                    rows.append(
                        {
                            "system": system_prompt,
                            "history": list(history),  # Copy
                            "response": content,
                            "tool_calls": msg.get("tool_calls"),
                            "agent_name": metadata.get("agent_name", "unknown"),
                            "task_type": metadata.get("task_type", "unknown"),
                        }
                    )
                    history.append({"role": "assistant", "content": content})
                elif role == "tool":
                    history.append({"role": "tool", "content": content})

        return Dataset.from_list(rows)

    raise ValueError(f"Unknown format: {format}. Use 'messages', 'chat', or 'turns'")
