"""
RolloutEnvironment protocol for async experience generation.

This module defines the protocol for async rollout generation in LLM training.
Unlike Gym environments, RolloutEnvironment generates complete trajectories
asynchronously rather than step-by-step interactions.

Key differences from Gym:
- Batch-oriented: Generates multiple completions per call
- Async-first: Designed for overlapped generation/training
- Complete trajectories: Returns full sequences, not per-step interactions
- LLM-specific: Handles tokenization, log probs, and reward computation

Usage:
    class VLLMRolloutEnv(RolloutEnvironment):
        async def generate_experiences(
            self,
            prompts: list[str],
            num_generations_per_prompt: int = 4,
        ) -> list[Experience]:
            # Generate with vLLM
            outputs = await self.llm.generate(prompts * num_generations_per_prompt)
            # Compute rewards
            rewards = await self.reward_fn(outputs)
            # Return experiences
            return [Experience(...) for output, reward in zip(outputs, rewards)]

    env = VLLMRolloutEnv(model="Qwen/Qwen2.5-1.5B-Instruct", reward_fn=my_reward)
    experiences = await env.generate_experiences(["What is 2+2?"])
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dreadnode.core.training.ray.experience import Experience, ExperienceBatch


@runtime_checkable
class RolloutEnvironment(Protocol):
    """
    Protocol for async rollout generation.

    This is the core abstraction for experience generation in async RL training.
    Implementations typically wrap vLLM or other inference engines and handle:
    - Batch generation of completions
    - Reward computation
    - Experience construction (token IDs, log probs, etc.)
    - Weight synchronization for on-policy training

    Unlike Gym environments:
    - No step() method - generates complete trajectories
    - No reset() - each call is independent
    - Batch-oriented for efficiency
    - Async-first design

    Implementations:
    - VLLMRolloutEnv: Uses vLLM for fast batch inference
    - MockRolloutEnv: For testing without GPU
    """

    @abstractmethod
    async def generate_experiences(
        self,
        prompts: list[str],
        num_generations_per_prompt: int = 4,
    ) -> list[Experience]:
        """
        Generate experiences for a batch of prompts.

        This is the core method for experience generation. For each prompt,
        generates `num_generations_per_prompt` completions and returns
        Experience objects with all necessary training data.

        Args:
            prompts: List of prompt strings
            num_generations_per_prompt: Number of completions per prompt (G in GRPO)

        Returns:
            List of Experience objects with:
            - Token IDs (prompt, completion, full)
            - Log probabilities from generation model
            - Rewards from reward function
            - Group IDs for GRPO baseline computation
        """
        ...

    @abstractmethod
    async def generate_batch(
        self,
        prompts: list[str],
        num_generations_per_prompt: int = 4,
    ) -> ExperienceBatch:
        """
        Generate an ExperienceBatch for a batch of prompts.

        Convenience method that wraps generate_experiences() and returns
        an ExperienceBatch ready for training.

        Args:
            prompts: List of prompt strings
            num_generations_per_prompt: Number of completions per prompt

        Returns:
            ExperienceBatch with all experiences and computed advantages
        """
        ...

    @abstractmethod
    def update_weights(self, state_dict: dict) -> None:
        """
        Update generation model weights (for on-policy training).

        Called periodically by the coordinator to sync the generation model
        with the training model. For off-policy training with high staleness,
        this can be called less frequently.

        Args:
            state_dict: Model state dict from the learner
        """
        ...


@runtime_checkable
class StreamingRolloutEnvironment(RolloutEnvironment, Protocol):
    """
    Extended protocol for streaming experience generation.

    Adds streaming capabilities for continuous generation, useful for
    async training where experiences are produced continuously.
    """

    @abstractmethod
    async def generate_stream(
        self,
        prompt_iterator: AsyncIterator[list[str]],
        num_generations_per_prompt: int = 4,
    ) -> AsyncIterator[Experience]:
        """
        Stream experiences continuously from prompts.

        Generates experiences in a streaming fashion, yielding individual
        experiences as they become available. Useful for async training
        where the learner consumes experiences as they are produced.

        Args:
            prompt_iterator: Async iterator yielding batches of prompts
            num_generations_per_prompt: Number of completions per prompt

        Yields:
            Individual Experience objects as they are generated
        """
        ...


@runtime_checkable
class MultiTurnRolloutEnvironment(RolloutEnvironment, Protocol):
    """
    Extended protocol for multi-turn conversation generation.

    Adds support for multi-turn conversations where the environment
    provides responses between model turns (e.g., tool use, agent tasks).
    """

    @abstractmethod
    async def generate_conversation(
        self,
        initial_prompts: list[str],
        max_turns: int = 5,
        num_generations_per_prompt: int = 4,
    ) -> list[Experience]:
        """
        Generate multi-turn conversation experiences.

        For each prompt, generates a multi-turn conversation where:
        1. Model generates a response
        2. Environment provides feedback (e.g., tool results)
        3. Repeat until termination or max_turns

        Args:
            initial_prompts: Initial prompts to start conversations
            max_turns: Maximum conversation turns
            num_generations_per_prompt: Number of parallel conversations per prompt

        Returns:
            List of Experience objects, one per complete conversation
        """
        ...

    @abstractmethod
    async def step(
        self,
        messages: list[dict],
    ) -> tuple[str, bool, dict]:
        """
        Execute a single environment step for multi-turn generation.

        Args:
            messages: Conversation history up to current point

        Returns:
            Tuple of (response, done, info)
            - response: Environment's response (e.g., tool output)
            - done: Whether conversation should terminate
            - info: Additional metadata
        """
        ...
