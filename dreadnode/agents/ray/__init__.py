"""
Shared agent components for Ray Actor integration.

This module provides the base classes and tools for building
LLM-powered Ray Actors across all services.

Trajectory utilities are re-exported from DN SDK for convenience:
- trajectory_to_openai_format: Convert trajectory to OpenAI messages
- trajectory_to_jsonl_record: Convert to training-ready JSONL record
"""

from core.agents.base import (
    AgentActorBase,
    EventFactory,
    trajectory_to_openai_format,
    trajectory_to_jsonl_record,
)
from core.agents.threaded import AgentHandoff, HandoffToolset
from core.agents.tools import (
    MessageBusToolset,
    create_bus_event_hook,
    log_generation_hook,
    log_tool_hook,
)

__all__ = [
    # Base
    "AgentActorBase",
    "EventFactory",
    # Trajectory utilities (from DN SDK)
    "trajectory_to_openai_format",
    "trajectory_to_jsonl_record",
    # Handoffs
    "AgentHandoff",
    "HandoffToolset",
    # Tools
    "MessageBusToolset",
    # Hooks
    "create_bus_event_hook",
    "log_generation_hook",
    "log_tool_hook",
]
