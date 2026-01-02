"""
Rollout orchestrator for multi-turn agent execution.

Matches NeMo RL's rollout pattern while integrating with DN SDK agents.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from dreadnode.core.training.rollouts.types import (
    Message,
    MessageLog,
    MessageRole,
    RolloutConfig,
    RolloutMetrics,
    RolloutResult,
    TurnResult,
)


@runtime_checkable
class GenerationInterface(Protocol):
    """
    Protocol for generation backends.

    Implemented by adapters for DN Agent, vLLM, OpenAI, etc.
    """

    async def generate(
        self,
        messages: MessageLog,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Generate a response given message history.

        Args:
            messages: Conversation history
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stop: Stop sequences

        Returns:
            Tuple of (generated_text, metadata)
            Metadata should include 'tool_calls' if any
        """
        ...


@dataclass
class EnvironmentReturn:
    """Return value from environment step."""

    observation: str
    """Environment observation/feedback"""

    reward: float = 0.0
    """Reward for this step"""

    terminated: bool = False
    """Whether episode is done (success or failure)"""

    truncated: bool = False
    """Whether episode was cut short (max turns, etc.)"""

    info: dict[str, Any] | None = None
    """Additional information"""


@runtime_checkable
class EnvironmentInterface(Protocol):
    """
    Protocol for training environments.

    Matches NeMo RL's EnvironmentInterface.
    """

    def step(
        self,
        message_logs: list[MessageLog],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[EnvironmentReturn]:
        """
        Process agent responses and return environment feedback.

        Args:
            message_logs: Batch of conversation histories
            metadata: Optional metadata per conversation

        Returns:
            List of EnvironmentReturn for each conversation
        """
        ...


class RolloutOrchestrator:
    """
    Orchestrates multi-turn rollouts for training.

    This class manages the agent-environment interaction loop,
    collecting trajectories suitable for RL training.

    Matches NeMo RL's rollout pattern:
    ```
    for turn in range(max_turns):
        response = policy.generate(message_log)
        message_log.append(assistant_msg)
        env_return = env.step(message_log)
        message_log.append(env_obs)
        if terminated: break
    ```

    Example:
        ```python
        from dreadnode import Agent
        from dreadnode.core.training import RolloutOrchestrator, AgentAdapter

        agent = Agent(model="...", tools=[...])
        adapter = AgentAdapter(agent)

        orchestrator = RolloutOrchestrator({"max_turns": 10})
        result = await orchestrator.run_single(adapter, "Do the task", env)
        ```
    """

    def __init__(
        self,
        config: RolloutConfig | None = None,
        tokenizer: Any | None = None,
    ):
        """
        Initialize orchestrator.

        Args:
            config: Rollout configuration
            tokenizer: Optional tokenizer for token counting
        """
        self.config = config or RolloutConfig(max_turns=10)
        self.tokenizer = tokenizer
        self.max_turns = self.config.get("max_turns", 10)
        self.max_tokens_per_turn = self.config.get("max_tokens_per_turn", 4096)
        self.stop_strings = self.config.get("stop_strings", [])
        self.timeout_per_turn = self.config.get("timeout_per_turn", 120.0)

    async def run_single(
        self,
        generator: GenerationInterface,
        goal: str,
        environment: EnvironmentInterface | None = None,
        system_prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RolloutResult:
        """
        Execute a single multi-turn rollout.

        Args:
            generator: Generation backend (agent adapter, vLLM, etc.)
            goal: The goal/prompt for the agent
            environment: Optional environment for feedback
            system_prompt: Optional system prompt
            metadata: Optional metadata to include in result

        Returns:
            RolloutResult with complete trajectory
        """
        rollout_id = str(uuid.uuid4())
        started_at = datetime.now()
        message_log: MessageLog = []
        turns: list[TurnResult] = []

        # Initialize message log
        if system_prompt:
            message_log.append(Message(role=MessageRole.SYSTEM.value, content=system_prompt))

        message_log.append(Message(role=MessageRole.USER.value, content=goal))

        # Metrics accumulators
        total_input_tokens = 0
        total_generated_tokens = 0
        total_reward = 0.0
        total_tool_calls = 0
        successful_tool_calls = 0
        failed_tool_calls = 0
        total_generation_time = 0.0
        total_env_time = 0.0

        terminated = False
        truncated = False
        errored = False
        success = False

        for turn_num in range(self.max_turns):
            turn_start = time.time()

            # Generate response
            try:
                gen_start = time.time()
                generated_text, gen_metadata = await asyncio.wait_for(
                    generator.generate(
                        message_log,
                        max_tokens=self.max_tokens_per_turn,
                        stop=self.stop_strings or None,
                    ),
                    timeout=self.timeout_per_turn,
                )
                gen_time = time.time() - gen_start
                total_generation_time += gen_time

            except asyncio.TimeoutError:
                errored = True
                turns.append(
                    TurnResult(
                        turn_number=turn_num,
                        generated_text="",
                        generated_tokens=0,
                        input_tokens=0,
                        error="Generation timeout",
                        terminated=True,
                    )
                )
                break

            except Exception as e:
                errored = True
                turns.append(
                    TurnResult(
                        turn_number=turn_num,
                        generated_text="",
                        generated_tokens=0,
                        input_tokens=0,
                        error=str(e),
                        terminated=True,
                    )
                )
                break

            # Extract tool calls from metadata
            tool_calls = gen_metadata.get("tool_calls", [])
            usage = gen_metadata.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            total_input_tokens += input_tokens
            total_generated_tokens += output_tokens
            total_tool_calls += len(tool_calls)

            # Add assistant message to log
            assistant_msg: Message = {
                "role": MessageRole.ASSISTANT.value,
                "content": generated_text,
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            message_log.append(assistant_msg)

            # Create turn result (will be updated with env response)
            turn_result = TurnResult(
                turn_number=turn_num,
                generated_text=generated_text,
                generated_tokens=output_tokens,
                input_tokens=input_tokens,
                tool_calls=tool_calls,
                generation_time=gen_time,
            )

            # Call environment if provided
            if environment is not None:
                env_start = time.time()
                try:
                    env_returns = environment.step([message_log], [metadata or {}])
                    env_return = env_returns[0]

                    env_time = time.time() - env_start
                    total_env_time += env_time
                    turn_result.env_time = env_time

                    # Add environment observation to message log
                    if env_return.observation:
                        message_log.append(
                            Message(
                                role=MessageRole.ENVIRONMENT.value,
                                content=env_return.observation,
                            )
                        )
                        turn_result.env_observation = env_return.observation

                    # Update rewards and status
                    turn_result.reward = env_return.reward
                    total_reward += env_return.reward

                    terminated = env_return.terminated
                    truncated = env_return.truncated
                    turn_result.terminated = terminated
                    turn_result.truncated = truncated

                    # Check for success in info
                    if env_return.info:
                        success = env_return.info.get("success", False)

                except Exception as e:
                    turn_result.error = f"Environment error: {e}"
                    errored = True
                    terminated = True

            # No environment - check for natural termination
            # (e.g., no tool calls means agent is done)
            elif not tool_calls and generated_text:
                terminated = True

            turns.append(turn_result)

            if terminated or truncated:
                break

        # Check if we hit max turns
        if not terminated and not truncated and len(turns) >= self.max_turns:
            truncated = True

        # Build metrics
        num_turns = len(turns)
        metrics = RolloutMetrics(
            total_turns=num_turns,
            completed_turns=sum(1 for t in turns if not t.error),
            total_input_tokens=total_input_tokens,
            total_generated_tokens=total_generated_tokens,
            mean_tokens_per_turn=(total_generated_tokens / num_turns if num_turns > 0 else 0),
            total_reward=total_reward,
            mean_reward_per_turn=total_reward / num_turns if num_turns > 0 else 0,
            final_reward=turns[-1].reward if turns else 0,
            natural_termination=terminated and not errored,
            truncated=truncated,
            errored=errored,
            total_time=(datetime.now() - started_at).total_seconds(),
            mean_generation_time=(total_generation_time / num_turns if num_turns > 0 else 0),
            mean_env_time=total_env_time / num_turns if num_turns > 0 else 0,
            total_tool_calls=total_tool_calls,
            successful_tool_calls=successful_tool_calls,
            failed_tool_calls=failed_tool_calls,
        )

        return RolloutResult(
            rollout_id=rollout_id,
            goal=goal,
            message_log=message_log,
            turns=turns,
            metrics=metrics,
            config=self.config,
            started_at=started_at,
            completed_at=datetime.now(),
            success=success,
            final_reward=total_reward,
            metadata=metadata or {},
        )

    async def run_batch(
        self,
        generator: GenerationInterface,
        goals: Sequence[str],
        environment: EnvironmentInterface | None = None,
        system_prompt: str | None = None,
        concurrency: int = 4,
    ) -> list[RolloutResult]:
        """
        Execute rollouts for a batch of goals.

        Args:
            generator: Generation backend
            goals: List of goals/prompts
            environment: Optional environment
            system_prompt: Optional system prompt
            concurrency: Max concurrent rollouts

        Returns:
            List of RolloutResults
        """
        semaphore = asyncio.Semaphore(concurrency)

        async def run_with_semaphore(goal: str) -> RolloutResult:
            async with semaphore:
                return await self.run_single(generator, goal, environment, system_prompt)

        results = await asyncio.gather(
            *[run_with_semaphore(goal) for goal in goals],
            return_exceptions=True,
        )

        # Handle exceptions
        final_results: list[RolloutResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                final_results.append(
                    RolloutResult(
                        rollout_id=str(uuid.uuid4()),
                        goal=goals[i],
                        metrics=RolloutMetrics(errored=True),
                        metadata={"error": str(result)},
                    )
                )
            else:
                final_results.append(result)

        return final_results

    async def run_batch_multi_turn(
        self,
        generator: GenerationInterface,
        goals: Sequence[str],
        environment: EnvironmentInterface,
        system_prompt: str | None = None,
    ) -> tuple[list[RolloutResult], dict[str, Any]]:
        """
        Execute batched multi-turn rollouts.

        This method processes all rollouts together, calling the environment
        once per turn for all active rollouts. More efficient for environments
        that can batch their processing.

        Matches NeMo RL's batched rollout pattern.

        Args:
            generator: Generation backend
            goals: List of goals/prompts
            environment: Environment (must support batched step)
            system_prompt: Optional system prompt

        Returns:
            Tuple of (results, aggregated_metrics)
        """
        batch_size = len(goals)
        results: list[RolloutResult] = []

        # Initialize all rollouts
        for goal in goals:
            rollout_id = str(uuid.uuid4())
            message_log: MessageLog = []

            if system_prompt:
                message_log.append(Message(role=MessageRole.SYSTEM.value, content=system_prompt))
            message_log.append(Message(role=MessageRole.USER.value, content=goal))

            results.append(
                RolloutResult(
                    rollout_id=rollout_id,
                    goal=goal,
                    message_log=message_log,
                    started_at=datetime.now(),
                )
            )

        # Track active rollouts
        active_mask = [True] * batch_size

        for turn_num in range(self.max_turns):
            # Get active indices
            active_indices = [i for i, active in enumerate(active_mask) if active]

            if not active_indices:
                break

            # Generate for all active rollouts
            generation_tasks = []
            for idx in active_indices:
                generation_tasks.append(
                    generator.generate(
                        results[idx].message_log,
                        max_tokens=self.max_tokens_per_turn,
                        stop=self.stop_strings or None,
                    )
                )

            gen_results = await asyncio.gather(*generation_tasks, return_exceptions=True)

            # Process generation results
            active_message_logs = []
            active_metadata = []

            for i, (idx, gen_result) in enumerate(zip(active_indices, gen_results)):
                if isinstance(gen_result, Exception):
                    active_mask[idx] = False
                    results[idx].metrics.errored = True
                    continue

                generated_text, gen_metadata = gen_result
                tool_calls = gen_metadata.get("tool_calls", [])

                # Add assistant message
                assistant_msg: Message = {
                    "role": MessageRole.ASSISTANT.value,
                    "content": generated_text,
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls

                results[idx].message_log.append(assistant_msg)
                active_message_logs.append(results[idx].message_log)
                active_metadata.append({})

            # Call environment for all active rollouts
            if active_message_logs:
                env_returns = environment.step(active_message_logs, active_metadata)

                # Process environment returns
                active_idx = 0
                for idx in active_indices:
                    if not active_mask[idx]:
                        continue

                    env_return = env_returns[active_idx]
                    active_idx += 1

                    if env_return.observation:
                        results[idx].message_log.append(
                            Message(
                                role=MessageRole.ENVIRONMENT.value,
                                content=env_return.observation,
                            )
                        )

                    results[idx].final_reward += env_return.reward

                    if env_return.terminated or env_return.truncated:
                        active_mask[idx] = False
                        results[idx].success = (
                            env_return.info.get("success", False) if env_return.info else False
                        )

        # Finalize results
        for result in results:
            result.completed_at = datetime.now()
            result.metrics.total_turns = len(
                [m for m in result.message_log if m["role"] == MessageRole.ASSISTANT.value]
            )

        # Aggregate metrics
        agg_metrics = {
            "total_rollouts": batch_size,
            "successful_rollouts": sum(1 for r in results if r.success),
            "mean_reward": sum(r.final_reward for r in results) / batch_size,
            "mean_turns": sum(r.metrics.total_turns for r in results) / batch_size,
        }

        return results, agg_metrics
