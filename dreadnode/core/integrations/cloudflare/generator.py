"""
Code generator for Cloudflare Workers.
"""

from __future__ import annotations

from importlib.resources import files

from jinja2 import BaseLoader, Environment

from dreadnode.core.integrations.serve.config import Serve


class _ResourceLoader(BaseLoader):
    """Jinja loader for importlib.resources."""

    def __init__(self, package: str):
        self.root = files(package)

    def get_source(self, environment: Environment, template: str) -> tuple[str, str, callable]:
        path = self.root.joinpath(template)
        source = path.read_text()
        return source, template, lambda: True


class WorkerGenerator:
    """
    Generates JavaScript Worker code from a Serve configuration.

    The generated worker handles:
    - Multiple components (Agents, Evaluations, Studies)
    - HTTP requests to configured endpoints
    - Queue message consumption
    - Authentication (API key, JWT, Cloudflare Access)
    - Request/response transformation
    - Error handling and retries
    """

    def __init__(self) -> None:
        self.env = Environment(
            loader=_ResourceLoader("dreadnode.core.integrations.cloudflare.templates"),
            autoescape=False,
        )

    def generate(
        self,
        serve: Serve,
        *,
        backend_url: str | None = None,
    ) -> str:
        """
        Generate JavaScript Worker code.

        Args:
            serve: The Serve configuration describing the components to serve.
            backend_url: URL of the Dreadnode Platform backend.

        Returns:
            JavaScript code for the Cloudflare Worker.
        """
        template = self.env.get_template("worker.js.j2")

        cors_origins = ", ".join(serve.cors_origins) if serve.cors_origins else "*"
        endpoint_paths = [e.path for e in serve.endpoints]

        return template.render(
            endpoints=serve.endpoints,
            queues=serve.queues,
            backend_url=backend_url or "{{BACKEND_URL}}",
            component_names=serve.component_names,
            endpoint_paths=endpoint_paths,
            required_secrets=serve.required_secrets,
            cors_origins=cors_origins,
        )

    def generate_wrangler(
        self,
        serve: Serve,
        worker_name: str,
        *,
        routes: list[str] | None = None,
    ) -> str:
        """
        Generate wrangler.toml configuration.

        Args:
            serve: The Serve configuration describing the components.
            worker_name: Name of the Cloudflare Worker.
            routes: Optional custom routes for the worker.

        Returns:
            TOML configuration for wrangler.
        """
        template = self.env.get_template("wrangler.toml.j2")

        return template.render(
            worker_name=worker_name,
            component_names=serve.component_names,
            queues=serve.queues,
            routes=routes or [],
            required_secrets=serve.required_secrets,
        )
