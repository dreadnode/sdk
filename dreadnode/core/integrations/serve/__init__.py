"""
Serve integration for exposing components as HTTP APIs.

This module provides the Serve class for configuring and running
Agents, Evaluations, and Studies as HTTP endpoints.

Example:
    ```python
    from dreadnode import Agent
    from dreadnode.core.integrations.serve import Serve

    # Create agents
    chat_agent = Agent(name="chat", model="openai/gpt-4", instructions="...")
    code_agent = Agent(name="code", model="anthropic/claude-3", instructions="...")

    # Configure and run server
    server = (
        Serve()
        .add(chat_agent, path="/chat")
        .add(code_agent, path="/code")
    )
    server.run(port=8000)
    ```
"""

from dreadnode.core.integrations.serve.config import (
    AuthMode,
    ComponentType,
    EndpointConfig,
    QueueConfig,
    Serve,
)

__all__ = [
    "AuthMode",
    "ComponentType",
    "EndpointConfig",
    "QueueConfig",
    "Serve",
]
