"""
Adapters for different generation backends.

Provides unified GenerationInterface for:
- DN SDK Agent
- vLLM HTTP server
- OpenAI API
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from dreadnode.core.training.rollouts.types import MessageLog

if TYPE_CHECKING:
    from dreadnode.core.agents import Agent


class DNAgentAdapter:
    """
    Adapter to use DN Agent with RolloutOrchestrator.

    This wraps a DN Agent to provide the GenerationInterface,
    allowing seamless integration with the training rollout system.

    Note: DN Agent has its own internal loop for tool execution.
    This adapter runs a single generation step, returning tool
    calls for external handling by the orchestrator.

    Example:
        ```python
        from dreadnode import Agent
        from dreadnode.core.training import RolloutOrchestrator, DNAgentAdapter

        agent = Agent(model="...", tools=[...])
        adapter = DNAgentAdapter(agent)

        orchestrator = RolloutOrchestrator(config)
        result = await orchestrator.run_single(adapter, goal, env)
        ```
    """

    def __init__(
        self,
        agent: "Agent",
        single_step: bool = True,
    ):
        """
        Initialize adapter.

        Args:
            agent: DN SDK Agent instance
            single_step: If True, only generate one response (for orchestrator control)
                        If False, let agent run its full loop
        """
        self.agent = agent
        self.single_step = single_step
        self._agent_id = getattr(agent, "agent_id", None)

    @property
    def tools(self) -> list[dict[str, Any]]:
        """Get tool schemas from agent."""
        if hasattr(self.agent, "all_tools"):
            return [t.api_definition for t in self.agent.all_tools]
        return []

    async def generate(
        self,
        messages: MessageLog,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate response using DN Agent.

        For single_step mode, we call the underlying model directly
        rather than using agent.run() which has its own loop.
        """
        if self.single_step:
            return await self._generate_single_step(messages, max_tokens, temperature, stop)
        return await self._generate_full_run(messages)

    async def _generate_single_step(
        self,
        messages: MessageLog,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate single step using agent's generator directly."""
        from dreadnode.core.generators.generator import GenerateParams
        from dreadnode.core.generators.message import Message

        try:
            # Access agent's generator property
            generator = self.agent.generator

            # Convert messages to Message format
            gen_messages = []
            for m in messages:
                gen_messages.append(
                    Message(role=m["role"], content=m.get("content", ""))
                )

            # Set up generation parameters
            params = GenerateParams(
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop or [],
            )

            # Get tool definitions if available
            tool_definitions = None
            if hasattr(self.agent, "all_tools") and self.agent.all_tools:
                tool_definitions = [t.api_definition for t in self.agent.all_tools]

            # Build chat pipeline
            pipeline = generator.chat(gen_messages, params=params)

            # Add tools if available
            if tool_definitions:
                pipeline = pipeline.using(*tool_definitions)

            # Run generation
            chat = await pipeline.run()

            # Extract response from last message
            last_message = chat.last
            text = last_message.content if last_message else ""
            tool_calls = []

            # Extract tool calls if present
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tc in last_message.tool_calls:
                    tool_calls.append(
                        {
                            "id": getattr(tc, "id", f"call_{len(tool_calls)}"),
                            "type": "function",
                            "function": {
                                "name": tc.name if hasattr(tc, "name") else str(tc),
                                "arguments": tc.arguments
                                if isinstance(getattr(tc, "arguments", ""), str)
                                else json.dumps(getattr(tc, "arguments", {})),
                            },
                        }
                    )

            # Get usage info if available
            usage = {}
            if hasattr(chat, "usage") and chat.usage:
                usage = {
                    "prompt_tokens": getattr(chat.usage, "input_tokens", 0),
                    "completion_tokens": getattr(chat.usage, "output_tokens", 0),
                }

            metadata = {
                "tool_calls": tool_calls,
                "usage": usage,
                "finish_reason": "stop" if not tool_calls else "tool_calls",
            }

            return text, metadata

        except Exception as e:
            # Return error info
            return "", {"error": str(e), "tool_calls": []}

    async def _generate_full_run(
        self,
        messages: MessageLog,
    ) -> tuple[str, dict[str, Any]]:
        """Run full agent loop and extract final response."""
        # Extract goal from last user message
        goal = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                goal = msg["content"]
                break

        if not goal:
            return "", {"error": "No user message found"}

        try:
            # Run agent (this has its own tool loop)
            trajectory = await self.agent.run(goal)

            # Extract final assistant response
            final_text = ""
            tool_calls: list[dict[str, Any]] = []
            tool_results: list[Any] = []

            for event in trajectory.events:
                if hasattr(event, "content"):
                    final_text = event.content or final_text
                if hasattr(event, "tool_calls"):
                    tool_calls.extend(event.tool_calls or [])
                if hasattr(event, "tool_result"):
                    tool_results.append(event.tool_result)

            metadata = {
                "trajectory": trajectory,
                "usage": {
                    "input_tokens": trajectory.usage.input_tokens
                    if hasattr(trajectory, "usage")
                    else 0,
                    "output_tokens": trajectory.usage.output_tokens
                    if hasattr(trajectory, "usage")
                    else 0,
                },
                "tool_calls": tool_calls,
                "tool_results": tool_results,
                "scores": dict(trajectory.scores) if hasattr(trajectory, "scores") else {},
            }

            return final_text, metadata

        except Exception as e:
            return "", {"error": str(e)}


class VllmAdapter:
    """
    Adapter for vLLM HTTP server.

    Uses httpx to communicate with vLLM's exposed HTTP server.

    Example:
        ```python
        from dreadnode.core.training import RolloutOrchestrator, VllmAdapter

        adapter = VllmAdapter(base_url="http://localhost:8000", model="trained-model")
        result = await orchestrator.run_single(adapter, goal, env)
        ```
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        model: str = "default",
        timeout: float = 120.0,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._client = None

    async def _get_client(self):
        """Lazy init of HTTP client."""
        if self._client is None:
            import httpx

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )
        return self._client

    async def generate(
        self,
        messages: MessageLog,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate using vLLM HTTP server."""
        client = await self._get_client()

        # Convert messages
        openai_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if stop:
            payload["stop"] = stop

        try:
            response = await client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            choice = data["choices"][0]
            text = choice["message"].get("content", "")
            tool_calls = choice["message"].get("tool_calls", [])

            metadata = {
                "usage": data.get("usage", {}),
                "tool_calls": tool_calls,
                "finish_reason": choice.get("finish_reason"),
            }

            return text, metadata

        except Exception as e:
            return "", {"error": str(e)}

    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
