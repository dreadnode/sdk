"""
Multi-turn rollout support for agent training with tools.

This module extends the rollout system with multi-turn conversation support,
enabling training of agents that use tools during inference.

Key components:
- MultiTurnConfig: Configuration for multi-turn rollouts
- ToolExecutor: Executes tools during rollouts
- MultiTurnRolloutWorker: Ray actor for multi-turn generation

Usage:
    from dreadnode.core.training.ray.multi_turn import (
        MultiTurnConfig,
        MultiTurnRolloutWorker,
    )

    # Define tools
    @tool
    def calculate(expression: str) -> str:
        return str(eval(expression))

    # Create worker
    worker = MultiTurnRolloutWorker.remote(
        config=MultiTurnConfig(
            model_name="Qwen/Qwen2.5-1.5B-Instruct",
            tools=[calculate],
            max_turns=5,
        ),
        reward_fn=my_reward_fn,
    )

    # Generate multi-turn conversations
    batch = ray.get(worker.generate_conversation.remote(prompts))
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import ray
import torch

from dreadnode.core.training.ray.config import RayGRPOConfig, VLLMConfig
from dreadnode.core.training.ray.experience import Experience, ExperienceBatch
from dreadnode.core.training.ray.callbacks import ToolCallInfo

if TYPE_CHECKING:
    from dreadnode.core.tools import Tool, ToolCall


@dataclass
class MultiTurnConfig:
    """Configuration for multi-turn rollout generation."""

    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    """Model name or path."""

    tokenizer_name: str | None = None
    """Tokenizer name (defaults to model_name)."""

    # Generation
    max_new_tokens: int = 512
    """Maximum tokens to generate per turn."""

    temperature: float = 0.7
    """Sampling temperature."""

    top_p: float = 0.9
    """Top-p (nucleus) sampling."""

    # Multi-turn
    max_turns: int = 5
    """Maximum conversation turns."""

    tools: list[Tool] = field(default_factory=list)
    """Tools available during generation."""

    stop_on_tool_error: bool = False
    """Whether to stop conversation on tool execution error."""

    # GRPO
    num_generations_per_prompt: int = 4
    """Number of completions per prompt."""

    use_leave_one_out_baseline: bool = True
    """Use leave-one-out baseline for advantages."""

    # vLLM
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    """vLLM configuration."""

    # Chat format
    system_prompt: str | None = None
    """Optional system prompt to prepend to all conversations."""

    def __post_init__(self) -> None:
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name

    @classmethod
    def from_grpo_config(
        cls,
        config: RayGRPOConfig,
        tools: list[Tool] | None = None,
        max_turns: int = 5,
    ) -> MultiTurnConfig:
        """Create from RayGRPOConfig."""
        return cls(
            model_name=config.model_name,
            tokenizer_name=config.tokenizer_name,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            num_generations_per_prompt=config.num_generations_per_prompt,
            use_leave_one_out_baseline=config.loss.use_leave_one_out_baseline,
            vllm=config.vllm,
            tools=tools or [],
            max_turns=max_turns,
        )


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: str
    """Role: 'user', 'assistant', or 'tool'."""

    content: str
    """Text content of the turn."""

    tool_calls: list[dict] | None = None
    """Tool calls made by assistant (if any)."""

    tool_call_id: str | None = None
    """ID of the tool call this is responding to (for tool role)."""

    logprobs: torch.Tensor | None = None
    """Log probabilities for this turn (assistant only)."""


@dataclass
class ConversationResult:
    """Result of a multi-turn conversation."""

    turns: list[ConversationTurn]
    """All turns in the conversation."""

    prompt: str
    """Initial prompt."""

    done: bool
    """Whether conversation ended naturally."""

    final_response: str
    """Final assistant response."""

    tool_calls_made: int
    """Total tool calls executed."""

    tool_errors: int
    """Number of tool execution errors."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""

    @property
    def full_text(self) -> str:
        """Get full conversation as text."""
        parts = []
        for turn in self.turns:
            if turn.role == "user":
                parts.append(f"User: {turn.content}")
            elif turn.role == "assistant":
                parts.append(f"Assistant: {turn.content}")
            elif turn.role == "tool":
                parts.append(f"Tool Result: {turn.content}")
        return "\n".join(parts)

    @property
    def all_logprobs(self) -> torch.Tensor:
        """Concatenate all logprobs from assistant turns."""
        lps = []
        for turn in self.turns:
            if turn.role == "assistant" and turn.logprobs is not None:
                lps.append(turn.logprobs)
        return torch.cat(lps) if lps else torch.tensor([])


class ToolExecutor:
    """
    Executes tools during multi-turn rollouts.

    Handles tool call parsing, execution, and error handling.
    """

    def __init__(
        self,
        tools: list[Tool],
        stop_on_error: bool = False,
    ) -> None:
        """
        Initialize tool executor.

        Args:
            tools: List of Tool objects
            stop_on_error: Whether to stop on tool execution errors
        """
        self.tools = {t.name: t for t in tools}
        self.stop_on_error = stop_on_error

        # Statistics
        self.calls_made = 0
        self.calls_succeeded = 0
        self.calls_failed = 0

    def get_tool_definitions(self) -> list[dict]:
        """Get tool definitions in OpenAI format for vLLM."""
        return [t.definition.model_dump() for t in self.tools.values()]

    def parse_tool_calls(self, text: str) -> list[dict]:
        """
        Parse tool calls from model output.

        Supports multiple formats:
        - JSON tool calls: {"name": "...", "arguments": {...}}
        - Function call syntax: tool_name(arg1, arg2)
        - XML format: <tool_call>...</tool_call>

        Args:
            text: Model output text

        Returns:
            List of parsed tool calls
        """
        tool_calls = []

        # Try JSON format first
        json_pattern = r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*"arguments"\s*:\s*(\{[^{}]*\})[^{}]*\}'
        for match in re.finditer(json_pattern, text, re.DOTALL):
            try:
                name = match.group(1)
                args = json.loads(match.group(2))
                if name in self.tools:
                    tool_calls.append({
                        "id": f"call_{uuid.uuid4().hex[:8]}",
                        "name": name,
                        "arguments": args,
                    })
            except json.JSONDecodeError:
                continue

        # Try function call syntax: tool_name(arg1="value", arg2=123)
        if not tool_calls:
            for tool_name in self.tools:
                pattern = rf'{tool_name}\s*\(([^)]*)\)'
                for match in re.finditer(pattern, text):
                    try:
                        args_str = match.group(1)
                        # Parse as Python-style kwargs
                        args = self._parse_function_args(args_str)
                        tool_calls.append({
                            "id": f"call_{uuid.uuid4().hex[:8]}",
                            "name": tool_name,
                            "arguments": args,
                        })
                    except Exception:
                        continue

        return tool_calls

    def _parse_function_args(self, args_str: str) -> dict:
        """Parse function-style arguments into a dict."""
        if not args_str.strip():
            return {}

        args = {}
        # Match key=value pairs (handles strings and numbers)
        pattern = r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|(\d+\.?\d*)|(\w+))'
        for match in re.finditer(pattern, args_str):
            key = match.group(1)
            # Get the first non-None captured value
            value = match.group(2) or match.group(3) or match.group(4) or match.group(5)
            # Try to convert to number if applicable
            if value and value.replace('.', '').isdigit():
                value = float(value) if '.' in str(value) else int(value)
            args[key] = value

        return args

    async def execute_tool(
        self,
        tool_call: dict,
    ) -> tuple[str, bool, ToolCallInfo]:
        """
        Execute a single tool call.

        Args:
            tool_call: Dict with 'name' and 'arguments'

        Returns:
            Tuple of (result_text, success, tool_info)
        """
        from dreadnode.core.tools import ToolCall as DnToolCall, FunctionCall

        name = tool_call["name"]
        args = tool_call["arguments"]
        call_id = tool_call.get("id", f"call_{uuid.uuid4().hex[:8]}")

        tool_info = ToolCallInfo(
            name=name,
            tool_call_id=call_id,
            arguments=json.dumps(args) if isinstance(args, dict) else str(args),
        )

        if name not in self.tools:
            tool_info.error = ValueError(f"Unknown tool: {name}")
            return f"Error: Unknown tool '{name}'", False, tool_info

        tool = self.tools[name]
        start_time = time.time()

        try:
            # Create ToolCall object
            dn_tool_call = DnToolCall(
                id=call_id,
                type="function",
                function=FunctionCall(
                    name=name,
                    arguments=json.dumps(args) if isinstance(args, dict) else str(args),
                ),
            )

            # Execute tool
            message, _stop = await tool.handle_tool_call(dn_tool_call)
            result = message.text if hasattr(message, 'text') else str(message.content)

            self.calls_made += 1
            self.calls_succeeded += 1

            tool_info.result = result
            tool_info.duration_seconds = time.time() - start_time

            return result, True, tool_info

        except Exception as e:
            self.calls_made += 1
            self.calls_failed += 1

            tool_info.error = e
            tool_info.duration_seconds = time.time() - start_time

            return f"Error executing {name}: {str(e)}", False, tool_info

    def execute_tool_sync(self, tool_call: dict) -> tuple[str, bool, ToolCallInfo]:
        """Synchronous wrapper for execute_tool."""
        return asyncio.get_event_loop().run_until_complete(
            self.execute_tool(tool_call)
        )

    def get_stats(self) -> dict:
        """Get tool execution statistics."""
        return {
            "calls_made": self.calls_made,
            "calls_succeeded": self.calls_succeeded,
            "calls_failed": self.calls_failed,
            "success_rate": self.calls_succeeded / self.calls_made if self.calls_made > 0 else 0,
        }


# Type alias for reward functions
RewardFunction = Callable[[list[str], list[str]], list[float]]


@ray.remote
class MultiTurnRolloutWorker:
    """
    Ray actor for multi-turn experience generation with tool support.

    Implements the MultiTurnRolloutEnvironment protocol for training
    agents that use tools during inference.
    """

    def __init__(
        self,
        config: MultiTurnConfig,
        reward_fn: RewardFunction,
        worker_id: int = 0,
        gpu_ids: list[int] | None = None,
    ) -> None:
        """
        Initialize multi-turn rollout worker.

        Args:
            config: Multi-turn configuration
            reward_fn: Reward function (prompts, completions) -> rewards
            worker_id: Unique worker ID
            gpu_ids: Specific GPU IDs to use
        """
        import os

        self.config = config
        self.reward_fn = reward_fn
        self.worker_id = worker_id

        # Set CUDA devices if specified
        if gpu_ids is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

        # Initialize vLLM
        self._init_vllm()

        # Initialize tool executor
        self.tool_executor = ToolExecutor(
            tools=config.tools,
            stop_on_error=config.stop_on_error,
        )

        # Statistics
        self._stats = {
            "conversations_generated": 0,
            "total_turns": 0,
            "weight_updates": 0,
        }

    def _init_vllm(self, model_path: str | None = None) -> None:
        """Initialize vLLM engine."""
        from vllm import LLM

        model = model_path or self.config.model_name

        self.llm = LLM(
            model=model,
            tensor_parallel_size=self.config.vllm.tensor_parallel_size,
            gpu_memory_utilization=self.config.vllm.gpu_memory_utilization,
            max_model_len=self.config.vllm.max_model_len,
            enforce_eager=self.config.vllm.enforce_eager,
            dtype=self.config.vllm.dtype,
            trust_remote_code=self.config.vllm.trust_remote_code,
            enable_prefix_caching=self.config.vllm.enable_prefix_caching,
        )
        self.tokenizer = self.llm.get_tokenizer()

    def generate_conversation(
        self,
        prompts: list[str],
        num_generations_per_prompt: int | None = None,
    ) -> ExperienceBatch:
        """
        Generate multi-turn conversation experiences.

        Args:
            prompts: Initial prompts
            num_generations_per_prompt: Completions per prompt

        Returns:
            ExperienceBatch with conversation experiences
        """
        from vllm import SamplingParams

        num_gen = num_generations_per_prompt or self.config.num_generations_per_prompt

        # Expand prompts
        expanded_prompts = []
        group_ids = []
        for i, prompt in enumerate(prompts):
            for _ in range(num_gen):
                expanded_prompts.append(prompt)
                group_ids.append(i)

        # Generate conversations
        experiences = []
        all_completions = []

        for idx, prompt in enumerate(expanded_prompts):
            result = self._generate_single_conversation(prompt)

            # Build experience from conversation
            full_text = result.full_text
            all_completions.append(full_text)

            # Tokenize for training
            prompt_ids = torch.tensor(
                self.tokenizer.encode(prompt),
                dtype=torch.long,
            )
            full_ids = torch.tensor(
                self.tokenizer.encode(prompt + "\n" + result.final_response),
                dtype=torch.long,
            )
            completion_ids = full_ids[len(prompt_ids):]

            experiences.append(
                Experience(
                    prompt=prompt,
                    completion=result.final_response,
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    logprobs=result.all_logprobs if len(result.all_logprobs) > 0 else torch.zeros(len(completion_ids)),
                    group_id=group_ids[idx],
                    reward=0.0,  # Set after reward computation
                    metadata={
                        "turns": len(result.turns),
                        "tool_calls": result.tool_calls_made,
                        "tool_errors": result.tool_errors,
                        "done": result.done,
                    },
                )
            )

        # Compute rewards
        all_prompts = [exp.prompt for exp in experiences]
        rewards = self.reward_fn(all_prompts, all_completions)

        for exp, reward in zip(experiences, rewards):
            exp.reward = reward

        # Create batch
        batch = ExperienceBatch(experiences=experiences)
        batch.compute_advantages(
            use_leave_one_out=self.config.use_leave_one_out_baseline
        )

        # Update stats
        self._stats["conversations_generated"] += len(expanded_prompts)
        self._stats["total_turns"] += sum(
            exp.metadata.get("turns", 0) for exp in experiences
        )

        return batch

    def _generate_single_conversation(self, prompt: str) -> ConversationResult:
        """Generate a single multi-turn conversation."""
        from vllm import SamplingParams

        turns: list[ConversationTurn] = []
        messages = []

        # Add system prompt if configured
        if self.config.system_prompt:
            messages.append({
                "role": "system",
                "content": self.config.system_prompt,
            })

        # Add user prompt
        messages.append({"role": "user", "content": prompt})
        turns.append(ConversationTurn(role="user", content=prompt))

        tool_calls_made = 0
        tool_errors = 0
        done = False
        final_response = ""

        for turn_idx in range(self.config.max_turns):
            # Format messages for model
            chat_text = self._format_messages(messages)

            # Generate
            sampling_params = SamplingParams(
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                logprobs=1,
            )

            outputs = self.llm.generate([chat_text], sampling_params)
            output = outputs[0].outputs[0]
            response_text = output.text
            logprobs = self._extract_logprobs(output)

            # Add assistant turn
            turns.append(ConversationTurn(
                role="assistant",
                content=response_text,
                logprobs=logprobs,
            ))
            messages.append({"role": "assistant", "content": response_text})
            final_response = response_text

            # Check for tool calls
            tool_calls = self.tool_executor.parse_tool_calls(response_text)

            if not tool_calls:
                # No tool calls - conversation is done
                done = True
                break

            # Execute tool calls
            for tc in tool_calls:
                result, success, tool_info = self.tool_executor.execute_tool_sync(tc)
                tool_calls_made += 1

                if not success:
                    tool_errors += 1
                    if self.config.stop_on_tool_error:
                        done = True
                        break

                # Add tool result
                turns.append(ConversationTurn(
                    role="tool",
                    content=result,
                    tool_call_id=tc.get("id"),
                ))
                messages.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": tc.get("id"),
                })

            if done:
                break

        return ConversationResult(
            turns=turns,
            prompt=prompt,
            done=done,
            final_response=final_response,
            tool_calls_made=tool_calls_made,
            tool_errors=tool_errors,
        )

    def _format_messages(self, messages: list[dict]) -> str:
        """Format messages for vLLM generation."""
        # Use tokenizer's chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        # Fallback to simple format
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            elif role == "tool":
                parts.append(f"Tool Result: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _extract_logprobs(self, completion: Any) -> torch.Tensor:
        """Extract log probabilities from vLLM completion."""
        if completion.logprobs is None:
            return torch.zeros(len(completion.token_ids))

        lps = []
        for lp_dict in completion.logprobs:
            if lp_dict:
                token_id = list(lp_dict.keys())[0]
                lp_obj = lp_dict[token_id]
                if hasattr(lp_obj, "logprob"):
                    lps.append(lp_obj.logprob)
                else:
                    lps.append(float(lp_obj))
            else:
                lps.append(0.0)

        return torch.tensor(lps, dtype=torch.float32)

    def step(
        self,
        messages: list[dict],
    ) -> tuple[str, bool, dict]:
        """
        Execute a single step (tool execution) for external coordination.

        Args:
            messages: Conversation history

        Returns:
            Tuple of (response, done, info)
        """
        # Get the last assistant message
        last_msg = messages[-1] if messages else None
        if not last_msg or last_msg.get("role") != "assistant":
            return "", True, {"error": "No assistant message to process"}

        # Parse and execute tool calls
        tool_calls = self.tool_executor.parse_tool_calls(last_msg.get("content", ""))

        if not tool_calls:
            return "", True, {"reason": "no_tool_calls"}

        # Execute first tool call
        tc = tool_calls[0]
        result, success, tool_info = self.tool_executor.execute_tool_sync(tc)

        return result, not success and self.config.stop_on_error, {
            "tool_name": tc["name"],
            "success": success,
            "tool_info": tool_info,
        }

    def update_weights(
        self,
        checkpoint_path: str | None = None,
        state_dict_ref: ray.ObjectRef | dict | None = None,
    ) -> bool:
        """Update vLLM weights."""
        import gc

        if checkpoint_path is not None:
            try:
                del self.llm
                gc.collect()
                torch.cuda.empty_cache()
                self._init_vllm(model_path=checkpoint_path)
                self._stats["weight_updates"] += 1
                return True
            except Exception as e:
                print(f"[Worker {self.worker_id}] Checkpoint reload failed: {e}")
                self._init_vllm()
                return False

        return False

    def get_stats(self) -> dict:
        """Get worker statistics."""
        stats = self._stats.copy()
        stats["tool_stats"] = self.tool_executor.get_stats()
        return stats

    def get_tool_definitions(self) -> list[dict]:
        """Get tool definitions for external use."""
        return self.tool_executor.get_tool_definitions()

    def shutdown(self) -> None:
        """Shutdown worker."""
        del self.llm
        torch.cuda.empty_cache()
