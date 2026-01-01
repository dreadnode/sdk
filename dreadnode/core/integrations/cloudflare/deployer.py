"""
Cloudflare Workers deployer for serving components.
"""

from __future__ import annotations

import typing as t

import httpx
from pydantic import BaseModel, Field, PrivateAttr

from dreadnode.core.integrations.serve.config import QueueConfig, Serve
from dreadnode.core.integrations.cloudflare.generator import WorkerGenerator


class DeploymentResult(BaseModel):
    """Result of a Cloudflare Workers deployment."""

    success: bool
    """Whether the deployment was successful."""

    worker_name: str
    """Name of the deployed worker."""

    worker_url: str | None = None
    """URL of the deployed worker (if available)."""

    routes: list[str] = Field(default_factory=list)
    """Custom routes attached to the worker."""

    components: list[str] = Field(default_factory=list)
    """Names of components being served."""

    endpoints: list[str] = Field(default_factory=list)
    """Endpoint paths configured on the worker."""

    errors: list[str] = Field(default_factory=list)
    """Any errors encountered during deployment."""

    warnings: list[str] = Field(default_factory=list)
    """Any warnings from the deployment."""


class CloudflareWorkersDeployer(BaseModel):
    """
    Deploys components to Cloudflare Workers via the Cloudflare API.

    This deployer generates JavaScript Worker code that proxies requests
    to the Dreadnode Platform where the components run.

    Example:
        ```python
        from dreadnode import Agent
        from dreadnode.core.integrations.cloudflare import Serve, CloudflareWorkersDeployer

        # Create agents
        chat_agent = Agent(name="chat", model="openai/gpt-4", ...)
        code_agent = Agent(name="code", model="anthropic/claude-3", ...)

        # Configure server with multiple agents
        server = (
            Serve()
            .add(chat_agent, path="/chat")
            .add(code_agent, path="/code")
        )

        # Deploy to Cloudflare Workers
        async with CloudflareWorkersDeployer(
            account_id="your-account-id",
            api_token="your-api-token",
        ) as deployer:
            result = await deployer.deploy(
                server,
                worker_name="my-agents",
                backend_url="https://platform.dreadnode.io/run",
            )
            print(f"Deployed to: {result.worker_url}")
        ```
    """

    account_id: str
    """Cloudflare account ID."""

    api_token: str = Field(repr=False)
    """Cloudflare API token with Workers permissions."""

    _client: httpx.AsyncClient | None = PrivateAttr(default=None)
    _generator: WorkerGenerator | None = PrivateAttr(default=None)

    @property
    def client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url="https://api.cloudflare.com/client/v4",
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json",
                },
                timeout=60,
            )
        return self._client

    @property
    def generator(self) -> WorkerGenerator:
        """Get or create the worker generator."""
        if self._generator is None:
            self._generator = WorkerGenerator()
        return self._generator

    async def deploy(
        self,
        serve: Serve,
        *,
        worker_name: str,
        backend_url: str,
        secrets: dict[str, str] | None = None,
        routes: list[str] | None = None,
    ) -> DeploymentResult:
        """
        Deploy a Serve configuration to Cloudflare Workers.

        Args:
            serve: The Serve configuration with components to deploy.
            worker_name: Name for the Worker.
            backend_url: URL of the Dreadnode Platform backend.
            secrets: Secrets to configure on the worker.
            routes: Custom routes to attach to the worker.

        Returns:
            DeploymentResult with deployment status and URLs.
        """
        result = DeploymentResult(
            success=False,
            worker_name=worker_name,
            components=serve.component_names,
            endpoints=[e.path for e in serve.endpoints],
        )

        try:
            # Generate worker code
            worker_code = self.generator.generate(serve, backend_url=backend_url)

            # Upload worker
            await self._upload_worker(worker_name, worker_code)

            # Configure backend URL and API key as secrets
            all_secrets = {
                "BACKEND_URL": backend_url,
                **(secrets or {}),
            }

            # Check for missing required secrets
            for secret_name in serve.required_secrets:
                if secret_name not in all_secrets:
                    result.warnings.append(f"Missing recommended secret: {secret_name}")

            # Configure secrets
            await self._configure_secrets(worker_name, all_secrets)

            # Configure routes if provided
            if routes:
                await self._configure_routes(worker_name, routes)
                result.routes = routes

            # Configure queues if any
            for queue_config in serve.queues:
                try:
                    await self.configure_queue(worker_name, queue_config)
                except Exception as e:
                    result.warnings.append(f"Failed to configure queue {queue_config.queue_name}: {e}")

            # Construct worker URL
            result.worker_url = f"https://{worker_name}.{self.account_id}.workers.dev"
            result.success = True

        except Exception as e:
            result.errors.append(str(e))

        return result

    async def _upload_worker(self, name: str, code: str) -> None:
        """Upload worker script to Cloudflare."""
        response = await self.client.put(
            f"/accounts/{self.account_id}/workers/scripts/{name}",
            content=code,
            headers={
                "Content-Type": "application/javascript",
            },
        )
        response.raise_for_status()

    async def _configure_secrets(self, worker_name: str, secrets: dict[str, str]) -> None:
        """Configure secrets on the worker."""
        for name, value in secrets.items():
            response = await self.client.put(
                f"/accounts/{self.account_id}/workers/scripts/{worker_name}/secrets",
                json={"name": name, "text": value, "type": "secret_text"},
            )
            # Secrets endpoint may return 409 if secret already exists, which is ok
            if response.status_code not in (200, 201, 409):
                response.raise_for_status()

    async def _configure_routes(self, worker_name: str, routes: list[str]) -> None:
        """Configure routes for the worker."""
        for route in routes:
            response = await self.client.post(
                f"/accounts/{self.account_id}/workers/routes",
                json={"pattern": route, "script": worker_name},
            )
            # Route may already exist
            if response.status_code not in (200, 201, 409):
                response.raise_for_status()

    async def configure_queue(
        self,
        worker_name: str,
        queue_config: QueueConfig,
    ) -> None:
        """
        Configure a queue binding for the worker.

        Args:
            worker_name: Name of the worker to bind the queue to.
            queue_config: Queue configuration.
        """
        # Create queue if it doesn't exist
        response = await self.client.post(
            f"/accounts/{self.account_id}/queues",
            json={"queue_name": queue_config.queue_name},
        )
        # Queue may already exist
        if response.status_code not in (200, 201, 409):
            response.raise_for_status()

        # Bind queue to worker as consumer
        settings: dict[str, t.Any] = {
            "batch_size": queue_config.batch_size,
            "max_retries": queue_config.max_retries,
        }
        if queue_config.dead_letter_queue:
            settings["dead_letter_queue"] = queue_config.dead_letter_queue

        response = await self.client.put(
            f"/accounts/{self.account_id}/workers/scripts/{worker_name}/queue-bindings/{queue_config.queue_name}",
            json={
                "queue_name": queue_config.queue_name,
                "type": "consumer",
                "settings": settings,
            },
        )
        if response.status_code not in (200, 201, 409):
            response.raise_for_status()

    async def delete(self, worker_name: str) -> bool:
        """
        Delete a deployed worker.

        Args:
            worker_name: Name of the worker to delete.

        Returns:
            True if deletion was successful.
        """
        response = await self.client.delete(
            f"/accounts/{self.account_id}/workers/scripts/{worker_name}"
        )
        return response.status_code == 200

    async def get_worker_url(self, worker_name: str) -> str:
        """
        Get the URL for a deployed worker.

        Args:
            worker_name: Name of the worker.

        Returns:
            The worker's URL.
        """
        return f"https://{worker_name}.{self.account_id}.workers.dev"

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "CloudflareWorkersDeployer":
        """Async context manager entry."""
        return self

    async def __aexit__(self, *args: t.Any) -> None:
        """Async context manager exit."""
        await self.close()
