"""
Cloudflare Workers integration for deploying components as serverless endpoints.

This module provides tools for deploying Dreadnode components to Cloudflare Workers.
Workers act as proxies to the Dreadnode Platform where components run.

Example:
    ```python
    from dreadnode import Agent
    from dreadnode.core.integrations.serve import Serve
    from dreadnode.core.integrations.cloudflare import CloudflareWorkersDeployer

    # Create agents
    chat_agent = Agent(name="chat", model="openai/gpt-4", instructions="...")
    code_agent = Agent(name="code", model="anthropic/claude-3", instructions="...")

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
        print(f"Deployed: {result.worker_url}")
    ```
"""

from dreadnode.core.integrations.cloudflare.deployer import (
    CloudflareWorkersDeployer,
    DeploymentResult,
)
from dreadnode.core.integrations.cloudflare.generator import WorkerGenerator

__all__ = [
    "CloudflareWorkersDeployer",
    "DeploymentResult",
    "WorkerGenerator",
]
