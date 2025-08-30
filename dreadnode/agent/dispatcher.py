import asyncio
import typing as t
from collections import defaultdict
from typing import Optional

from loguru import logger
from pydantic import BaseModel

if t.TYPE_CHECKING:
    from dreadnode.agent.agent import Agent


class AgentDispatcher:
    """
    Manages agent registration and message routing.
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}
        self._subscriptions: dict[type[BaseModel], set[str]] = defaultdict(set)
        logger.info("Dispatcher started.")

    async def register_agent(self, agent: "Agent"):
        """
        Register an agent with the dispatcher.
        Automatically subscribes the agent to message types based on its handlers."""
        if agent.unique_name in self._agents:
            logger.warning(
                f"Agent with name '{agent.unique_name}' is already registered. Overwriting."
            )
        self._agents[agent.unique_name] = agent

        handler = getattr(agent.__class__, "handle_message", None)
        if handler and hasattr(handler, "_handled_types"):
            for msg_type in handler._handled_types:
                self._subscriptions[msg_type].add(agent.unique_name)
                logger.info(f"Agent '{agent.unique_name}' subscribed to {msg_type.__name__}")

    def get_agent(self, name: str) -> Optional["Agent"]:
        """Get an agent proxy by name."""
        return self._agents.get(name)

    async def remove_agent(self, name: str):
        """Remove an agent from the dispatcher."""
        if name in self._agents:
            agent_instance = self._agents.get(name)
            if agent_instance:
                handler = getattr(agent_instance.__class__, "handle_message", None)
                if handler and hasattr(handler, "_handled_types"):
                    for msg_type in handler._handled_types:
                        if msg_type in self._subscriptions:
                            self._subscriptions[msg_type].discard(name)
            del self._agents[name]
            logger.info(f"Agent '{name}' removed from dispatcher.")

    def list_agents(self) -> list[str]:
        """List all registered agent names."""
        return list(self._agents.keys())

    def get_subscribers(self, message_type: type[BaseModel]) -> list["Agent"]:
        """
        Get all agents subscribed to a specific message type.
        """
        agent_names = self._subscriptions.get(message_type, set())
        return [self._agents[name] for name in agent_names if name in self._agents]

    async def publish(self, message: Dispatchable):
        """
        Publish a message to all agents subscribed to its type (fire-and-forget).
        """
        message_type = type(message)
        message_data_type = type(message.data)
        data_subscribers = self.get_subscribers(message_data_type)
        message_subscribers = self.get_subscribers(message_type)
        subscribers = data_subscribers + message_subscribers

        if not subscribers:
            logger.warning(f"No subscribers for message type {message_type.__name__}")
            return 0

        for agent in subscribers:
            agent.tell(message)
        logger.info(f"Published '{message_type.__name__}' to {len(subscribers)} agents.")
        return len(subscribers)

    async def shutdown_all(self):
        """Shutdown all registered agents."""
        logger.info("Shutting down all agents.")
        agents = list(self._agents.values())
        await asyncio.gather(*[agent.shutdown() for agent in agents], return_exceptions=True)
        self._agents.clear()
        self._subscriptions.clear()
        logger.info("All agents shut down and dispatcher cleared.")
