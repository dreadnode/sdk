"""
Ray Serve deployer for scaling components.
"""

from __future__ import annotations

import typing as t

from pydantic import BaseModel, Field

if t.TYPE_CHECKING:
    from ray.serve import Application

    from dreadnode.core.agents.agent import Agent
    from dreadnode.core.integrations.serve import Serve


class RayServeDeployer(BaseModel):
    """
    Deploy components using Ray Serve for distributed scaling.

    Ray Serve provides:
    - Horizontal scaling with multiple replicas
    - Autoscaling based on load
    - Fractional GPU allocation
    - Production-ready serving infrastructure

    Example:
        ```python
        from dreadnode import Agent
        from dreadnode.core.integrations.serve import Serve
        from dreadnode.core.integrations.ray import RayServeDeployer

        # Option 1: Deploy from Serve configuration
        server = Serve().add(agent, path="/chat")
        deployer = RayServeDeployer()
        deployer.run(server, num_replicas=2)

        # Option 2: Deploy from installed packages
        deployer.run_from_packages(
            "agent://org/chat-agent",
            "agent://org/code-agent",
            num_replicas=2,
        )
        ```
    """

    ray_address: str | None = Field(default=None)
    """Ray cluster address. None for local cluster."""

    namespace: str = "serve"
    """Ray namespace for deployments."""

    def deploy(
        self,
        serve: Serve,
        *,
        name: str = "dreadnode",
        num_replicas: int = 1,
        max_concurrent_queries: int = 100,
        ray_actor_options: dict[str, t.Any] | None = None,
        autoscaling_config: dict[str, t.Any] | None = None,
    ) -> Application:
        """
        Deploy a Serve configuration to Ray Serve.

        Args:
            serve: The Serve configuration with components to deploy.
            name: Name for the Ray Serve application.
            num_replicas: Number of replicas for each deployment.
            max_concurrent_queries: Max concurrent requests per replica.
            ray_actor_options: Ray actor options (num_cpus, num_gpus, etc.).
            autoscaling_config: Autoscaling configuration.

        Returns:
            Ray Serve Application handle.
        """
        import ray
        from ray import serve as ray_serve

        if not ray.is_initialized():
            ray.init(address=self.ray_address, namespace=self.namespace)

        app = serve.app(title=name)

        deployment_options: dict[str, t.Any] = {
            "num_replicas": num_replicas,
            "max_concurrent_queries": max_concurrent_queries,
        }

        if ray_actor_options:
            deployment_options["ray_actor_options"] = ray_actor_options

        if autoscaling_config:
            deployment_options["autoscaling_config"] = autoscaling_config

        deployment = ray_serve.ingress(app)
        ray_app = deployment.options(**deployment_options).bind()

        return ray_serve.run(ray_app, name=name, route_prefix="/")

    def deploy_from_packages(
        self,
        *packages: str,
        name: str = "dreadnode",
        num_replicas: int = 1,
        max_concurrent_queries: int = 100,
        ray_actor_options: dict[str, t.Any] | None = None,
        autoscaling_config: dict[str, t.Any] | None = None,
        pull: bool = True,
    ) -> Application:
        """
        Deploy agents from installed packages to Ray Serve.

        This method:
        1. Optionally pulls packages from the registry
        2. Discovers agents from entry points
        3. Creates a Serve configuration
        4. Deploys to Ray Serve

        Args:
            *packages: Package URIs (e.g., "agent://org/my-agent") or names.
            name: Name for the Ray Serve application.
            num_replicas: Number of replicas.
            max_concurrent_queries: Max concurrent requests per replica.
            ray_actor_options: Ray actor options.
            autoscaling_config: Autoscaling configuration.
            pull: Whether to pull packages before deploying.

        Returns:
            Ray Serve Application handle.

        Example:
            ```python
            deployer = RayServeDeployer()
            deployer.deploy_from_packages(
                "agent://acme/chat-agent",
                "agent://acme/code-agent",
                num_replicas=4,
            )
            ```
        """
        from dreadnode.core.integrations.ray.loader import discover_agents, load_agent
        from dreadnode.core.integrations.serve import Serve

        # Pull packages if requested
        if pull and packages:
            self._pull_packages(list(packages))

        # Build Serve configuration from discovered agents
        serve = Serve()

        if packages:
            # Load specific agents
            for pkg in packages:
                agent_name = self._parse_package_name(pkg)
                try:
                    agent = load_agent(agent_name)
                    path = f"/{agent_name.replace('.', '/')}"
                    serve.add(agent, path=path)
                except KeyError:
                    print(f"Warning: Agent not found: {agent_name}")
        else:
            # Discover all installed agents
            agents = discover_agents()
            for agent_name, agent in agents.items():
                path = f"/{agent_name.replace('.', '/')}"
                serve.add(agent, path=path)

        return self.deploy(
            serve,
            name=name,
            num_replicas=num_replicas,
            max_concurrent_queries=max_concurrent_queries,
            ray_actor_options=ray_actor_options,
            autoscaling_config=autoscaling_config,
        )

    def run(
        self,
        serve: Serve,
        *,
        name: str = "dreadnode",
        num_replicas: int = 1,
        max_concurrent_queries: int = 100,
        ray_actor_options: dict[str, t.Any] | None = None,
        autoscaling_config: dict[str, t.Any] | None = None,
        host: str = "0.0.0.0",
        port: int = 8000,
    ) -> None:
        """
        Deploy and run Ray Serve (blocking).

        Args:
            serve: The Serve configuration with components to deploy.
            name: Name for the Ray Serve application.
            num_replicas: Number of replicas.
            max_concurrent_queries: Max concurrent requests per replica.
            ray_actor_options: Ray actor options.
            autoscaling_config: Autoscaling configuration.
            host: Host to bind HTTP server to.
            port: Port to bind HTTP server to.
        """
        import ray
        from ray import serve as ray_serve

        if not ray.is_initialized():
            ray.init(address=self.ray_address, namespace=self.namespace)

        ray_serve.start(http_options={"host": host, "port": port})

        self.deploy(
            serve,
            name=name,
            num_replicas=num_replicas,
            max_concurrent_queries=max_concurrent_queries,
            ray_actor_options=ray_actor_options,
            autoscaling_config=autoscaling_config,
        )

        self._block_forever(host, port)

    def run_from_packages(
        self,
        *packages: str,
        name: str = "dreadnode",
        num_replicas: int = 1,
        max_concurrent_queries: int = 100,
        ray_actor_options: dict[str, t.Any] | None = None,
        autoscaling_config: dict[str, t.Any] | None = None,
        host: str = "0.0.0.0",
        port: int = 8000,
        pull: bool = True,
    ) -> None:
        """
        Deploy agents from packages and run Ray Serve (blocking).

        Args:
            *packages: Package URIs or names.
            name: Name for the Ray Serve application.
            num_replicas: Number of replicas.
            max_concurrent_queries: Max concurrent requests per replica.
            ray_actor_options: Ray actor options.
            autoscaling_config: Autoscaling configuration.
            host: Host to bind HTTP server to.
            port: Port to bind HTTP server to.
            pull: Whether to pull packages before deploying.

        Example:
            ```python
            deployer = RayServeDeployer()
            deployer.run_from_packages(
                "agent://acme/chat-agent",
                num_replicas=4,
                ray_actor_options={"num_gpus": 0.5},
            )
            ```
        """
        import ray
        from ray import serve as ray_serve

        if not ray.is_initialized():
            ray.init(address=self.ray_address, namespace=self.namespace)

        ray_serve.start(http_options={"host": host, "port": port})

        self.deploy_from_packages(
            *packages,
            name=name,
            num_replicas=num_replicas,
            max_concurrent_queries=max_concurrent_queries,
            ray_actor_options=ray_actor_options,
            autoscaling_config=autoscaling_config,
            pull=pull,
        )

        self._block_forever(host, port)

    def discover_agents(self) -> dict[str, Agent]:
        """
        Discover all installed agent packages.

        Returns:
            Dictionary mapping agent names to Agent instances.
        """
        from dreadnode.core.integrations.ray.loader import discover_agents

        return discover_agents()

    def shutdown(self) -> None:
        """Shutdown Ray Serve and Ray."""
        import ray
        from ray import serve as ray_serve

        ray_serve.shutdown()
        if ray.is_initialized():
            ray.shutdown()

    def _pull_packages(self, packages: list[str]) -> None:
        """Pull packages from registry."""
        # Filter to only agent:// URIs
        agent_packages = [p for p in packages if p.startswith("agent://")]

        if not agent_packages:
            return

        try:
            from dreadnode.core.packaging.package import Package
            from dreadnode.core.storage.storage import Storage

            # Get storage with session
            storage = Storage.default()
            Package.pull(*agent_packages, _storage=storage)
        except Exception as e:
            print(f"Warning: Failed to pull packages: {e}")

    def _parse_package_name(self, uri: str) -> str:
        """Parse a package URI to get the entry point name."""
        if "://" in uri:
            # agent://org/name -> org.name
            _, path = uri.split("://", 1)
            return path.replace("/", ".")
        return uri

    def _block_forever(self, host: str, port: int) -> None:
        """Block forever, handling shutdown signals."""
        import signal
        import time

        import ray
        from ray import serve as ray_serve

        print(f"Ray Serve running at http://{host}:{port}")
        print("Press Ctrl+C to stop...")

        def signal_handler(sig: int, frame: t.Any) -> None:
            print("\nShutting down Ray Serve...")
            ray_serve.shutdown()
            ray.shutdown()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        while True:
            time.sleep(1)
