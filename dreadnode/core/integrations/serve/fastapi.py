"""
FastAPI integration for serving components.
"""

from __future__ import annotations

import typing as t

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from dreadnode.core.integrations.serve.config import AuthMode, ComponentType, Serve

if t.TYPE_CHECKING:
    from dreadnode.core.agents.agent import Agent


class AgentRequest(BaseModel):
    """Request body for agent execution."""

    goal: str


class EvaluationRequest(BaseModel):
    """Request body for evaluation execution."""

    inputs: list[dict[str, t.Any]] | None = None
    dataset: list[dict[str, t.Any]] | None = None


class StudyRequest(BaseModel):
    """Request body for study execution."""

    parameters: dict[str, t.Any]
    max_trials: int | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    components: list[str]
    endpoints: list[str]


class EndpointInfo(BaseModel):
    """Information about an endpoint."""

    path: str
    method: str
    component: str
    type: str


class EndpointsResponse(BaseModel):
    """List of available endpoints."""

    endpoints: list[EndpointInfo]


def create_app(serve: Serve, *, title: str | None = None) -> FastAPI:
    """
    Create a FastAPI application from a Serve configuration.

    Args:
        serve: The Serve configuration with components to expose.
        title: Optional title for the API.

    Returns:
        A configured FastAPI application.
    """
    app_title = title or "Dreadnode API"
    if serve.component_names:
        app_title = f"{app_title} - {', '.join(serve.component_names)}"

    app = FastAPI(
        title=app_title,
        description="API for Dreadnode components (Agents, Evaluations, Studies)",
        version="1.0.0",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=serve.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store serve config in app state
    app.state.serve = serve

    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            components=serve.component_names,
            endpoints=[e.path for e in serve.endpoints],
        )

    # List endpoints
    @app.get("/", response_model=EndpointsResponse)
    async def list_endpoints() -> EndpointsResponse:
        return EndpointsResponse(
            endpoints=[
                EndpointInfo(
                    path=e.path,
                    method=e.method,
                    component=e.name,
                    type=e.component_type.value,
                )
                for e in serve.endpoints
            ]
        )

    # Create endpoint handlers for each configured endpoint
    for endpoint in serve.endpoints:
        _create_endpoint_handler(app, serve, endpoint)

    return app


def _create_endpoint_handler(
    app: FastAPI,
    serve: Serve,
    endpoint: t.Any,
) -> None:
    """Create a route handler for an endpoint."""
    component = serve.get_component(endpoint.name)
    if component is None:
        return

    # Create auth dependency based on auth mode
    auth_dependency = _create_auth_dependency(endpoint.auth_mode)

    if endpoint.component_type == ComponentType.AGENT:
        _create_agent_endpoint(app, endpoint.path, component, auth_dependency)
    elif endpoint.component_type == ComponentType.EVALUATION:
        _create_evaluation_endpoint(app, endpoint.path, component, auth_dependency)
    elif endpoint.component_type == ComponentType.STUDY:
        _create_study_endpoint(app, endpoint.path, component, auth_dependency)


def _create_auth_dependency(
    auth_mode: AuthMode,
) -> t.Callable[..., t.Awaitable[None]] | None:
    """Create an authentication dependency based on auth mode."""
    if auth_mode == AuthMode.NONE:
        return None

    async def verify_api_key(
        x_api_key: str | None = Header(None, alias="X-API-Key"),
        authorization: str | None = Header(None),
    ) -> None:
        api_key = x_api_key
        if not api_key and authorization:
            if authorization.startswith("Bearer "):
                api_key = authorization[7:]

        if not api_key:
            raise HTTPException(
                status_code=401,
                detail="Missing API key. Provide X-API-Key header or Authorization: Bearer <key>",
            )

    if auth_mode == AuthMode.API_KEY:
        return verify_api_key

    return None


def _create_agent_endpoint(
    app: FastAPI,
    path: str,
    agent: "Agent",
    auth_dependency: t.Callable[..., t.Awaitable[None]] | None,
) -> None:
    """Create an endpoint for an Agent."""
    dependencies = [Depends(auth_dependency)] if auth_dependency else []

    @app.post(path, dependencies=dependencies, tags=["agents"])
    async def run_agent(request: AgentRequest) -> dict[str, t.Any]:
        """Execute the agent with a goal and return the trajectory."""
        trajectory = await agent.run(request.goal)
        return trajectory.model_dump()


def _create_evaluation_endpoint(
    app: FastAPI,
    path: str,
    evaluation: t.Any,
    auth_dependency: t.Callable[..., t.Awaitable[None]] | None,
) -> None:
    """Create an endpoint for an Evaluation."""
    dependencies = [Depends(auth_dependency)] if auth_dependency else []

    @app.post(path, dependencies=dependencies, tags=["evaluations"])
    async def run_evaluation(request: EvaluationRequest) -> dict[str, t.Any]:
        """Execute the evaluation and return results."""
        if request.inputs or request.dataset:
            eval_instance = evaluation.clone()
            if request.inputs:
                eval_instance.dataset = request.inputs
            elif request.dataset:
                eval_instance.dataset = request.dataset
        else:
            eval_instance = evaluation

        result = await eval_instance.run()
        return result.model_dump()


def _create_study_endpoint(
    app: FastAPI,
    path: str,
    study: t.Any,
    auth_dependency: t.Callable[..., t.Awaitable[None]] | None,
) -> None:
    """Create an endpoint for a Study."""
    dependencies = [Depends(auth_dependency)] if auth_dependency else []

    @app.post(path, dependencies=dependencies, tags=["studies"])
    async def run_study(request: StudyRequest) -> dict[str, t.Any]:
        """Execute the optimization study and return results."""
        study_instance = study.clone()
        result = await study_instance.run()
        return result.model_dump()
