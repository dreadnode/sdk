"""
Ray Serve integration for scaling and deploying components.

This module provides tools for deploying Dreadnode components using Ray Serve,
enabling distributed scaling and production deployments.

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
    deployer = RayServeDeployer()
    deployer.run_from_packages(
        "agent://org/chat-agent",
        "agent://org/code-agent",
        num_replicas=4,
    )

    # Option 3: Discover and deploy all installed agents
    deployer = RayServeDeployer()
    agents = deployer.discover_agents()
    print(f"Found agents: {list(agents.keys())}")
    deployer.run_from_packages(num_replicas=2)  # Deploy all
    ```
"""

from dreadnode.core.integrations.ray.deployer import RayServeDeployer
from dreadnode.core.integrations.ray.loader import discover_agents, load_agent

__all__ = [
    "RayServeDeployer",
    "discover_agents",
    "load_agent",
]
