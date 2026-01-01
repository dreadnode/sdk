"""
Agent environment wrapper for NeMo RL GRPO training.

This environment evaluates model-generated agent responses by:
1. Parsing tool calls from responses
2. Optionally executing tools (online mode with vLLM HTTP server)
3. Computing rewards based on task success

Supports two modes:
- **Offline**: Compare generated responses to reference trajectories
- **Online**: Execute tool calls against live targets
"""

from typing import Any, NotRequired, TypedDict
import json
import re

import ray
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn


class AgentEnvConfig(TypedDict):
    """Configuration for agent environment."""

    num_workers: int
    """Number of reward computation workers."""

    # Reward configuration
    success_reward: float
    """Reward for successful task completion."""

    failure_reward: float
    """Reward for failure."""

    partial_reward: float
    """Reward for partial progress."""

    # Evaluation mode
    eval_mode: NotRequired[str]
    """'offline' (compare to reference) or 'online' (execute tools)."""

    # Stop strings for generation
    stop_strings: NotRequired[list[str] | None]

    # Online mode configuration
    vllm_base_url: NotRequired[str]
    """vLLM HTTP server URL for online mode."""

    target_host: NotRequired[str]
    """Target for tool execution."""

    execution_timeout: NotRequired[float]
    """Timeout for tool execution (seconds)."""

    max_tool_calls: NotRequired[int]
    """Maximum tool calls per response."""


class AgentEnvMetadata(TypedDict):
    """Metadata for agent environment."""

    reference_trajectory: list[dict[str, Any]] | None
    """Reference trajectory for offline evaluation."""

    expected_success: bool | None
    """Expected success indicator."""

    task_type: str
    """Type of task."""

    agent_name: str
    """Name of agent."""

    extra: dict[str, Any]
    """Additional context."""


@ray.remote
class AgentRewardWorker:
    """
    Worker for computing agent rewards.

    Evaluates generated responses against reference trajectories or
    executes them in a sandbox for online evaluation.
    """

    def __init__(self, config: AgentEnvConfig):
        self.config = config
        self.success_reward = config.get("success_reward", 1.0)
        self.failure_reward = config.get("failure_reward", 0.0)
        self.partial_reward = config.get("partial_reward", 0.5)
        self.eval_mode = config.get("eval_mode", "offline")
        self.max_tool_calls = config.get("max_tool_calls", 10)

    def compute_rewards(
        self,
        responses: list[str],
        metadata_batch: list[AgentEnvMetadata],
    ) -> tuple[list[float], list[bool], list[str | None]]:
        """
        Compute rewards for a batch of responses.

        Args:
            responses: Generated model responses.
            metadata_batch: Metadata including reference trajectories.

        Returns:
            Tuple of (rewards, terminateds, extracted_answers).
        """
        rewards = []
        terminateds = []
        extracted_answers: list[str | None] = []

        for response, metadata in zip(responses, metadata_batch):
            reward, terminated, answer = self._evaluate_single(response, metadata)
            rewards.append(reward)
            terminateds.append(terminated)
            extracted_answers.append(answer)

        return rewards, terminateds, extracted_answers

    def _evaluate_single(
        self,
        response: str,
        metadata: AgentEnvMetadata,
    ) -> tuple[float, bool, str | None]:
        """Evaluate a single response."""
        if self.eval_mode == "offline":
            return self._offline_evaluate(response, metadata)
        elif self.eval_mode == "online":
            return self._online_evaluate(response, metadata)
        else:
            return self._offline_evaluate(response, metadata)

    def _online_evaluate(
        self,
        response: str,
        metadata: AgentEnvMetadata,
    ) -> tuple[float, bool, str | None]:
        """Online evaluation by executing tool calls."""
        tool_calls = self._parse_tool_calls(response)

        if not tool_calls:
            if len(response) > 50 and any(
                kw in response.lower() for kw in ["let me", "i'll", "first"]
            ):
                return self.partial_reward * 0.5, False, "reasoning_only"
            return self.failure_reward, True, "no_tools"

        total_reward = 0.0
        executed_count = 0
        last_result = None

        for tc in tool_calls[: self.max_tool_calls]:
            try:
                result = self._execute_tool(tc)
                if result.get("success"):
                    total_reward += self.partial_reward
                    executed_count += 1
                    last_result = result.get("output", "")

                    if self._check_task_complete(result, metadata):
                        return self.success_reward, True, "task_complete"
                else:
                    total_reward += self.partial_reward * 0.3
            except Exception as e:
                last_result = f"error: {e}"

        if executed_count > 0:
            avg_reward = total_reward / executed_count
            return avg_reward, False, last_result
        else:
            return self.failure_reward, True, "execution_failed"

    def _parse_tool_calls(self, response: str) -> list[dict[str, Any]]:
        """Parse tool calls from response text."""
        tool_calls = []

        # Pattern 1: JSON tool call format
        json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*[^{}]*\}'
        for match in re.finditer(json_pattern, response):
            try:
                tc = json.loads(match.group())
                if "name" in tc:
                    tool_calls.append(
                        {
                            "name": tc["name"],
                            "arguments": tc.get("arguments", tc.get("parameters", {})),
                        }
                    )
            except json.JSONDecodeError:
                continue

        # Pattern 2: OpenAI function call format
        openai_pattern = r'"function"\s*:\s*\{[^}]+\}'
        for match in re.finditer(openai_pattern, response):
            try:
                tc = json.loads("{" + match.group() + "}")
                func = tc.get("function", {})
                if "name" in func:
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    tool_calls.append({"name": func["name"], "arguments": args})
            except json.JSONDecodeError:
                continue

        return tool_calls

    def _execute_tool(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Execute a single tool call (stub for real implementation)."""
        name = tool_call.get("name", "")
        args = tool_call.get("arguments", {})

        known_safe_tools = {
            "nmap_scan",
            "netexec_enum",
            "get_domain_info",
            "list_shares",
            "check_credentials",
            "bloodhound_collect",
        }

        if name in known_safe_tools:
            return {
                "success": True,
                "output": f"Executed {name} with {args}",
                "tool": name,
            }
        else:
            return {
                "success": False,
                "error": f"Unknown tool: {name}",
                "tool": name,
            }

    def _check_task_complete(
        self,
        result: dict[str, Any],
        metadata: AgentEnvMetadata,
    ) -> bool:
        """Check if the task is complete based on execution result."""
        task_type = metadata.get("task_type", "")
        output = str(result.get("output", ""))

        if "recon" in task_type.lower():
            return "domain" in output.lower() and "dc" in output.lower()
        elif "credential" in task_type.lower():
            return "valid" in output.lower() or "success" in output.lower()
        elif "lateral" in task_type.lower():
            return "connected" in output.lower() or "session" in output.lower()

        return False

    def _offline_evaluate(
        self,
        response: str,
        metadata: AgentEnvMetadata,
    ) -> tuple[float, bool, str | None]:
        """Offline evaluation by comparing to reference trajectory."""
        reference = metadata.get("reference_trajectory")
        expected_success = metadata.get("expected_success")

        if expected_success is not None:
            reward = self._compute_trajectory_similarity(response, reference)
            if reward > 0.8:
                return self.success_reward, True, "matched"
            elif reward > 0.5:
                return self.partial_reward, True, "partial"
            else:
                return self.failure_reward, True, "mismatch"

        if self._has_valid_tool_calls(response):
            return self.partial_reward, True, "valid_structure"
        else:
            return self.failure_reward, True, "invalid"

    def _compute_trajectory_similarity(
        self,
        response: str,
        reference: list[dict[str, Any]] | None,
    ) -> float:
        """Compute similarity between response and reference trajectory."""
        if not reference:
            return 0.5

        response_tools = self._extract_tool_names(response)

        reference_tools = []
        for msg in reference:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if isinstance(tc, dict) and "function" in tc:
                        reference_tools.append(tc["function"].get("name", ""))

        if not reference_tools:
            return 0.5

        if not response_tools:
            return 0.0

        overlap = len(set(response_tools) & set(reference_tools))
        union = len(set(response_tools) | set(reference_tools))

        return overlap / union if union > 0 else 0.0

    def _extract_tool_names(self, response: str) -> list[str]:
        """Extract tool names from response text."""
        pattern1 = r'"name"\s*:\s*"([^"]+)"'
        pattern2 = r"(\w+)\s*\("

        names = []
        names.extend(re.findall(pattern1, response))
        for match in re.findall(pattern2, response):
            if "_" in match or match.islower():
                names.append(match)

        return names

    def _has_valid_tool_calls(self, response: str) -> bool:
        """Check if response contains valid tool call structure."""
        try:
            json_pattern = r'\{[^{}]*"name"[^{}]*\}'
            matches = re.findall(json_pattern, response)
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if "name" in parsed:
                        return True
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass

        tool_names = self._extract_tool_names(response)
        return len(tool_names) > 0


@ray.remote(max_restarts=-1, max_task_retries=-1)
class AgentEnvironment(EnvironmentInterface[AgentEnvMetadata]):
    """
    NeMo RL environment for agent training.

    Evaluates model-generated agent responses and computes rewards.

    Usage:
        env = AgentEnvironment.remote(config)
        result = ray.get(env.step.remote(message_log_batch, metadata))
    """

    def __init__(self, cfg: AgentEnvConfig):
        self.cfg = cfg
        self.num_workers = cfg["num_workers"]

        self.workers = [
            AgentRewardWorker.remote(cfg) for _ in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        """Shutdown all workers."""
        for worker in self.workers:
            ray.kill(worker)

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[AgentEnvMetadata],
        return_extracted_answer: bool = False,
    ) -> EnvironmentReturn[AgentEnvMetadata]:
        """
        Evaluate a batch of agent responses.

        Args:
            message_log_batch: Batch of conversation histories.
            metadata: Batch of metadata with reference trajectories.
            return_extracted_answer: Whether to return extracted answers.

        Returns:
            EnvironmentReturn with observations, rewards, etc.
        """
        responses = []
        for conversation in message_log_batch:
            assistant_responses = [
                str(msg.get("content", ""))
                for msg in conversation
                if msg.get("role") == "assistant"
            ]
            responses.append("\n".join(assistant_responses))

        batch_size = len(responses)
        chunk_size = max(1, batch_size // self.num_workers)

        futures = []
        for i in range(0, batch_size, chunk_size):
            chunk_responses = responses[i : i + chunk_size]
            chunk_metadata = metadata[i : i + chunk_size]
            worker_idx = min(i // chunk_size, len(self.workers) - 1)
            futures.append(
                self.workers[worker_idx].compute_rewards.remote(
                    chunk_responses, chunk_metadata
                )
            )

        results = ray.get(futures)

        all_rewards = []
        all_terminateds = []
        all_answers: list[str | None] = []

        for rewards, terminateds, answers in results:
            all_rewards.extend(rewards)
            all_terminateds.extend(terminateds)
            all_answers.extend(answers)

        observations = [
            {
                "role": "environment",
                "content": f"Reward: {r:.2f}" if r > 0 else "Task not completed",
            }
            for r in all_rewards
        ]

        rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32)
        terminateds_tensor = torch.tensor(all_terminateds, dtype=torch.bool)

        stop_strings = self.cfg.get("stop_strings")
        next_stop_strings = (
            [stop_strings] * batch_size if stop_strings else [None] * batch_size
        )

        return EnvironmentReturn(
            observations=observations,
            metadata=metadata,
            next_stop_strings=next_stop_strings,
            rewards=rewards_tensor,
            terminateds=terminateds_tensor,
            answers=all_answers if return_extracted_answer else None,
        )

    def global_post_process_and_metrics(
        self,
        batch: BatchedDataDict[Any],
    ) -> tuple[BatchedDataDict[Any], dict[str, float | int]]:
        """Compute global metrics for the batch."""
        rewards = batch.get("rewards", torch.tensor([]))
        is_end = batch.get("is_end", torch.ones_like(rewards))

        batch["rewards"] = rewards * is_end

        metrics = {
            "mean_reward": rewards.mean().item() if len(rewards) > 0 else 0.0,
            "success_rate": (rewards > 0.5).float().mean().item()
            if len(rewards) > 0
            else 0.0,
            "num_samples": len(rewards),
            "fraction_properly_ended": is_end.float().mean().item()
            if len(is_end) > 0
            else 0.0,
        }

        return batch, metrics
