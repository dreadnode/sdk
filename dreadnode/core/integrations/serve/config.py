"""
Configuration classes for serving components.
"""

from __future__ import annotations

import typing as t
from enum import Enum

from pydantic import BaseModel, Field, PrivateAttr

if t.TYPE_CHECKING:
    from fastapi import FastAPI

    from dreadnode.core.agents.agent import Agent
    from dreadnode.core.evaluations.evaluation import Evaluation
    from dreadnode.core.optimization.study import Study


class AuthMode(str, Enum):
    """Authentication mode for the server endpoint."""

    NONE = "none"
    API_KEY = "api_key"
    JWT = "jwt"


class ComponentType(str, Enum):
    """Type of component being served."""

    AGENT = "agent"
    EVALUATION = "evaluation"
    STUDY = "study"


class EndpointConfig(BaseModel):
    """Configuration for an HTTP endpoint."""

    path: str
    """HTTP path for the endpoint."""

    name: str
    """Name of the component being served at this endpoint."""

    component_type: ComponentType
    """Type of component (agent, evaluation, study)."""

    method: t.Literal["POST"] = "POST"
    """HTTP method (only POST supported for execution)."""

    auth_mode: AuthMode = AuthMode.API_KEY
    """Authentication mode for the endpoint."""

    timeout_seconds: int = 300
    """Request timeout in seconds."""


class QueueConfig(BaseModel):
    """Configuration for queue consumption."""

    queue_name: str
    """Name of the queue to consume from."""

    component_name: str
    """Name of the component that processes queue messages."""

    component_type: ComponentType
    """Type of component (agent, evaluation, study)."""

    batch_size: int = 1
    """Number of messages to process per batch."""

    max_retries: int = 3
    """Maximum retry attempts for failed messages."""


class Serve(BaseModel):
    """
    Configure multiple components (Agents, Evaluations, Studies) for serving.

    This class wraps one or more components and exposes them as HTTP endpoints
    via FastAPI.

    Example:
        ```python
        from dreadnode import Agent
        from dreadnode.core.integrations.serve import Serve

        # Create agents
        chat_agent = Agent(name="chat", model="openai/gpt-4", ...)
        code_agent = Agent(name="code", model="anthropic/claude-3", ...)

        # Configure server with multiple agents
        server = (
            Serve()
            .add(chat_agent, path="/chat")
            .add(code_agent, path="/code")
        )

        # Run the server
        server.run(port=8000)

        # Or get the FastAPI app for custom configuration
        app = server.app()
        ```
    """

    _components: dict[str, t.Any] = PrivateAttr(default_factory=dict)
    """Stored components by name."""

    endpoints: list[EndpointConfig] = Field(default_factory=list)
    """HTTP endpoint configurations."""

    queues: list[QueueConfig] = Field(default_factory=list)
    """Queue consumer configurations."""

    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    """Allowed CORS origins."""

    auth_mode: AuthMode = AuthMode.API_KEY
    """Default authentication mode for endpoints."""

    def add(
        self,
        component: "Agent | Evaluation | Study",
        *,
        path: str | None = None,
        queue: str | None = None,
        name: str | None = None,
        auth_mode: AuthMode | None = None,
    ) -> "Serve":
        """
        Add a component to be served.

        Args:
            component: An Agent, Evaluation, or Study to serve.
            path: HTTP endpoint path (e.g., "/chat"). Required for HTTP access.
            queue: Queue name for async processing.
            name: Override the component name. Defaults to component's name attribute.
            auth_mode: Override authentication mode for this endpoint.

        Returns:
            Self for chaining.

        Example:
            ```python
            server = (
                Serve()
                .add(agent1, path="/agent1")
                .add(agent2, path="/agent2", queue="agent2-tasks")
            )
            ```
        """
        component_name = name or getattr(component, "name", None)
        if not component_name:
            raise ValueError("Component must have a name or provide one via 'name' parameter")

        # Determine component type
        component_type = self._get_component_type(component)

        # Store component reference
        self._components[component_name] = component

        # Add HTTP endpoint if path provided
        if path:
            self.endpoints.append(
                EndpointConfig(
                    path=path,
                    name=component_name,
                    component_type=component_type,
                    auth_mode=auth_mode or self.auth_mode,
                )
            )

        # Add queue consumer if queue provided
        if queue:
            self.queues.append(
                QueueConfig(
                    queue_name=queue,
                    component_name=component_name,
                    component_type=component_type,
                )
            )

        return self

    def with_auth(self, auth_mode: AuthMode) -> "Serve":
        """
        Set default authentication mode for all endpoints.

        Args:
            auth_mode: Authentication mode to use.

        Returns:
            Self for chaining.
        """
        self.auth_mode = auth_mode
        return self

    def with_cors(self, *origins: str) -> "Serve":
        """
        Set allowed CORS origins.

        Args:
            *origins: Allowed origins (e.g., "https://example.com").

        Returns:
            Self for chaining.
        """
        self.cors_origins = list(origins)
        return self

    def get_component(self, name: str) -> t.Any:
        """Get a component by name."""
        return self._components.get(name)

    @property
    def component_names(self) -> list[str]:
        """Get list of all component names."""
        return list(self._components.keys())

    def _get_component_type(self, component: t.Any) -> ComponentType:
        """Determine the type of a component."""
        from dreadnode.core.agents.agent import Agent

        class_name = type(component).__name__

        if class_name == "Agent" or isinstance(component, Agent):
            return ComponentType.AGENT
        elif class_name == "Evaluation":
            return ComponentType.EVALUATION
        elif class_name == "Study":
            return ComponentType.STUDY
        else:
            raise ValueError(f"Unsupported component type: {class_name}")

    def app(self, *, title: str | None = None) -> "FastAPI":
        """
        Create a FastAPI application that serves all configured components.

        Args:
            title: Optional title for the FastAPI app.

        Returns:
            A FastAPI application instance.

        Example:
            ```python
            server = Serve().add(agent, path="/chat")
            app = server.app()

            # Run with uvicorn directly
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000)
            ```
        """
        from dreadnode.core.integrations.serve.fastapi import create_app

        return create_app(self, title=title)

    def run(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        reload: bool = False,
        workers: int = 1,
        title: str | None = None,
    ) -> None:
        """
        Run the server with uvicorn.

        Args:
            host: Host to bind to.
            port: Port to bind to.
            reload: Enable auto-reload on code changes.
            workers: Number of worker processes.
            title: Optional title for the API.

        Example:
            ```python
            server = (
                Serve()
                .add(chat_agent, path="/chat")
                .add(code_agent, path="/code")
            )
            server.run(port=8000)
            ```
        """
        import uvicorn

        app = self.app(title=title)
        uvicorn.run(
            app,
            host=host,
            port=port,
            reload=reload,
            workers=workers,
        )
