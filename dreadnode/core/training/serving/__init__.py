"""
Model serving utilities for training.

Provides clients and utilities for vLLM HTTP server integration.

Key components:
- VllmClient: Client for vLLM's OpenAI-compatible API
- VllmTrainingContext: Context manager for training with vLLM
- create_openai_client: Create OpenAI client for vLLM
- wait_for_vllm_ready: Wait for vLLM server to start

Usage:
    from dreadnode.core.training import VllmClient, create_openai_client

    # Create client for vLLM server
    async with VllmClient(base_url="http://localhost:8000") as client:
        response = await client.chat_completion(messages)

    # Or create OpenAI client
    openai_client = create_openai_client("http://localhost:8000")
"""

from dreadnode.core.training.serving.vllm_client import (
    VllmClient,
    VllmTrainingContext,
    create_openai_client,
    wait_for_vllm_ready,
)

__all__ = [
    "VllmClient",
    "VllmTrainingContext",
    "create_openai_client",
    "wait_for_vllm_ready",
]
