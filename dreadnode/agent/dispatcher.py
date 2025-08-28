import typing as t
from typing import Optional

from loguru import logger

from dreadnode.agent.events import Event

if t.TYPE_CHECKING:
    from dreadnode.agent.agent import Agent
    from dreadnode.agent.types import Message, ToolCall, Usage


class Dispatcher:
    def on_agent_start(self, agent: "Agent") -> Event:
        logger.info(f"Agent started: {agent.name}")

    def on_step_start(self, step: int) -> Event:
        logger.info(f"Step {step} started")

    def on_generation_end(self, message: "Message", usage: Optional["Usage"]) -> Event:
        logger.info(f"Generation ended with message: {message.content}")

    def on_tool_start(self, tool_call: "ToolCall") -> Event:
        logger.info(f"Tool started: {tool_call.name}")

    def on_tool_end(self, tool_call: "ToolCall", message: "Message", stop: bool) -> Event:
        logger.info(f"Tool ended: {tool_call.name} with message: {message.content}")

    def on_agent_stalled(self, agent: "Agent") -> Event:
        logger.warning("Agent has stalled")

    def on_agent_error(self, error: Exception) -> Event:
        logger.error(f"Agent encountered an error: {error}")

    def on_agent_end(self, agent: "Agent") -> Event:
        logger.info(f"{agent.name} has completed its run")

    def catch(self, callback, *args, **kwargs):
        try:
            return callback(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {callback.__qualname__}(): {e}")
