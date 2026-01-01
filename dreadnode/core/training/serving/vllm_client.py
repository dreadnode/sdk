"""
vLLM HTTP client for agent training integration.

Provides a client for communicating with vLLM's exposed HTTP server
during NeMo RL training. When vLLM is configured with `expose_http_server: true`,
it provides an OpenAI-compatible API that agents can use.

Usage:
    # During training, get the vLLM server URL from the environment
    client = VllmClient(base_url="http://localhost:8000")

    # Use with DN Agent
    agent = Agent(
        model="openai/trained-model",
        client=OpenAI(base_url=client.base_url),
        ...
    )
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx


@dataclass
class VllmClient:
    """
    Client for vLLM's OpenAI-compatible HTTP server.

    Wraps httpx for async HTTP calls to vLLM, providing convenient
    methods for agent-style interactions.

    Example:
        async with VllmClient(base_url="http://localhost:8000") as client:
            response = await client.chat_completion(messages)
            print(response["choices"][0]["message"]["content"])
    """

    base_url: str = "http://localhost:8000"
    """Base URL of vLLM HTTP server."""

    timeout: float = 120.0
    """Request timeout in seconds."""

    model: str = "default"
    """Model ID to use."""

    _client: httpx.AsyncClient | None = field(default=None, repr=False)

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the underlying HTTP client."""
        if self._client is None:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        return self._client

    @property
    def openai_base_url(self) -> str:
        """Get the OpenAI-compatible base URL."""
        return f"{self.base_url}/v1"

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stop: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Send a chat completion request.

        Args:
            messages: OpenAI-format messages.
            tools: Tool definitions for function calling.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            stop: Stop sequences.

        Returns:
            OpenAI-format completion response.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = tools

        if stop:
            payload["stop"] = stop

        response = await self.client.post(
            "/v1/chat/completions",
            json=payload,
        )
        response.raise_for_status()
        return response.json()

    async def chat_completion_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Stream a chat completion request.

        Yields:
            OpenAI-format streaming chunks.
        """
        import json

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if tools:
            payload["tools"] = tools

        async with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    yield json.loads(data)

    async def tokenize(self, text: str) -> dict[str, Any]:
        """
        Tokenize text using vLLM's tokenizer.

        Args:
            text: Text to tokenize.

        Returns:
            Token IDs and count.
        """
        response = await self.client.post(
            "/tokenize",
            json={"text": text},
        )
        response.raise_for_status()
        return response.json()

    async def health_check(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            response = await self.client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    async def get_model_info(self) -> dict[str, Any]:
        """Get model information from vLLM."""
        response = await self.client.get("/v1/models")
        response.raise_for_status()
        return response.json()


def create_openai_client(vllm_base_url: str) -> Any:
    """
    Create an OpenAI client configured for vLLM.

    This can be used directly with DN Agent:

        client = create_openai_client("http://localhost:8000")
        agent = Agent(model="trained-model", client=client, ...)

    Args:
        vllm_base_url: Base URL of vLLM HTTP server.

    Returns:
        OpenAI client configured for vLLM.
    """
    try:
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            base_url=f"{vllm_base_url}/v1",
            api_key="not-needed",  # vLLM doesn't require API key
        )
    except ImportError:
        raise ImportError(
            "OpenAI package required. Install with: pip install openai"
        )


async def wait_for_vllm_ready(
    base_url: str = "http://localhost:8000",
    timeout: float = 300.0,
    poll_interval: float = 5.0,
) -> bool:
    """
    Wait for vLLM HTTP server to become ready.

    Useful at the start of training to ensure the model is loaded.

    Args:
        base_url: vLLM server URL.
        timeout: Maximum time to wait.
        poll_interval: Time between health checks.

    Returns:
        True if server became ready, False if timeout.
    """
    import time

    start = time.time()

    async with VllmClient(base_url=base_url) as client:
        while time.time() - start < timeout:
            if await client.health_check():
                return True
            await asyncio.sleep(poll_interval)

    return False


@dataclass
class VllmTrainingContext:
    """
    Context manager for training with vLLM HTTP server.

    Provides easy access to vLLM endpoints during training:

        async with VllmTrainingContext(dp_urls) as ctx:
            # Get client for a specific data parallel shard
            client = ctx.get_client(dp_idx=0)
            response = await client.chat_completion(messages)

    Example:
        dp_urls = ["http://localhost:8000", "http://localhost:8001"]
        async with VllmTrainingContext(dp_urls) as ctx:
            for i in range(ctx.dp_size):
                client = ctx.get_client(i)
                print(await client.health_check())
    """

    dp_base_urls: list[str]
    """One URL per data parallel shard."""

    timeout: float = 120.0
    """Request timeout."""

    _clients: list[VllmClient] = field(default_factory=list, repr=False)

    async def __aenter__(self):
        self._clients = [
            VllmClient(base_url=url, timeout=self.timeout)
            for url in self.dp_base_urls
        ]
        for client in self._clients:
            await client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        for client in self._clients:
            await client.__aexit__(exc_type, exc_val, exc_tb)
        self._clients = []

    def get_client(self, dp_idx: int = 0) -> VllmClient:
        """Get client for a specific data parallel shard."""
        return self._clients[dp_idx % len(self._clients)]

    @property
    def dp_size(self) -> int:
        """Number of data parallel shards."""
        return len(self._clients)

    async def broadcast_chat(
        self,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Send same request to all DP shards (for validation).

        Returns responses from all shards.
        """
        tasks = [
            client.chat_completion(messages, **kwargs) for client in self._clients
        ]
        return await asyncio.gather(*tasks)
