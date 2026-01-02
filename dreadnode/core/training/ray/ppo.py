"""
PPO (Proximal Policy Optimization) trainer for RLHF.

This module provides PPO training for language models, the classic RLHF algorithm
used in InstructGPT and many other systems.

PPO uses:
- A policy (actor) network that generates responses
- A value (critic) network that estimates expected returns
- GAE (Generalized Advantage Estimation) for variance reduction
- Clipped surrogate objective for stable updates

The PPO-Clip objective is:
    L = min(r * A, clip(r, 1-eps, 1+eps) * A)

where r = pi(a|s) / pi_old(a|s) and A is the GAE advantage.

References:
- https://arxiv.org/abs/1707.06347 (PPO paper)
- https://arxiv.org/abs/2203.02155 (InstructGPT)
- https://github.com/OpenRLHF/OpenRLHF

Usage:
    from dreadnode.core.training.ray.ppo import PPOTrainer, PPOConfig

    config = PPOConfig(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        reward_model_name="my-reward-model",
    )

    trainer = PPOTrainer(config)
    trainer.train(prompts)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from dreadnode.core.training.ray.fsdp2_learner import FSDP2Config

if TYPE_CHECKING:
    from collections.abc import Callable

    from dreadnode.core.storage.storage import Storage
    from dreadnode.models.local import LocalModel


@dataclass
class PPOConfig:
    """Configuration for PPO training."""

    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    """Policy model name or path."""

    tokenizer_name: str | None = None
    """Tokenizer name (defaults to model_name)."""

    # Reward model (optional - can use reward_fn instead)
    reward_model_name: str | None = None
    """Reward model name or path. If None, must provide reward_fn to train()."""

    # PPO hyperparameters
    clip_ratio: float = 0.2
    """PPO clipping ratio (epsilon)."""

    value_clip_ratio: float = 0.2
    """Value function clipping ratio."""

    kl_coef: float = 0.1
    """KL penalty coefficient."""

    kl_target: float | None = 0.01
    """Target KL divergence. If exceeded, KL coef is increased."""

    entropy_coef: float = 0.01
    """Entropy bonus coefficient."""

    gamma: float = 1.0
    """Discount factor (1.0 for episodic tasks like text generation)."""

    gae_lambda: float = 0.95
    """GAE lambda for advantage estimation."""

    # Sequence settings
    max_seq_length: int = 2048
    """Maximum sequence length."""

    max_new_tokens: int = 512
    """Maximum new tokens to generate."""

    # Generation
    temperature: float = 0.7
    """Sampling temperature."""

    top_p: float = 0.9
    """Top-p sampling."""

    # Training
    learning_rate: float = 1e-6
    """Learning rate for policy."""

    critic_lr: float = 1e-5
    """Learning rate for value function (typically higher than policy)."""

    weight_decay: float = 0.01
    """Weight decay."""

    warmup_ratio: float = 0.1
    """Warmup steps as fraction of total."""

    max_steps: int = 1000
    """Maximum training steps."""

    batch_size: int = 8
    """Prompts per batch."""

    mini_batch_size: int = 4
    """Mini-batch size for PPO updates."""

    ppo_epochs: int = 4
    """Number of PPO epochs per batch of experience."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm."""

    # Reference model
    ref_model_offload: bool = True
    """Keep reference model on CPU to save GPU memory."""

    # Critic
    share_critic: bool = False
    """Share weights between policy and critic (adds value head to policy)."""

    critic_warmup_steps: int = 0
    """Pretrain critic for N steps before PPO (0 = no warmup)."""

    # Logging & Checkpointing
    log_interval: int = 10
    """Steps between logging."""

    checkpoint_interval: int = 100
    """Steps between checkpoints."""

    checkpoint_dir: str = "./checkpoints"
    """Directory for checkpoints."""

    # Seed
    seed: int = 42
    """Random seed."""

    # Trust remote code
    trust_remote_code: bool = True
    """Trust remote code in model repository."""

    def __post_init__(self) -> None:
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name


class ValueHead(nn.Module):
    """Value head for critic network."""

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            Values: [batch_size, seq_len]
        """
        x = self.dropout(hidden_states)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        values = self.out_proj(x).squeeze(-1)
        return values


class PPOTrainer:
    """
    PPO (Proximal Policy Optimization) trainer for RLHF.

    Implements the full PPO algorithm with:
    - Policy network (actor)
    - Value network (critic)
    - GAE advantage estimation
    - Clipped surrogate objective
    - KL penalty and adaptive KL coefficient

    The training loop:
    1. Generate responses from current policy
    2. Compute rewards using reward model/function
    3. Estimate advantages with GAE
    4. Update policy and value networks with PPO

    Attributes:
        config: PPO configuration
        policy: Policy (actor) model
        critic: Value (critic) model
        ref_model: Frozen reference model for KL penalty
        tokenizer: Tokenizer
    """

    def __init__(
        self,
        config: PPOConfig,
        fsdp_config: FSDP2Config | None = None,
        storage: Storage | None = None,
        checkpoint_name: str | None = None,
    ) -> None:
        """
        Initialize PPO trainer.

        Args:
            config: PPO configuration
            fsdp_config: Optional FSDP2 configuration
            storage: Optional storage for CAS checkpointing
            checkpoint_name: Name for checkpoints
        """
        self.config = config
        self.fsdp_config = fsdp_config or FSDP2Config(
            ref_model_offload=config.ref_model_offload
        )
        self.storage = storage
        self.checkpoint_name = checkpoint_name or config.model_name.replace("/", "-")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._checkpoint_version = 0
        self.step = 0
        self.kl_coef = config.kl_coef  # Adaptive KL coefficient

        # Load models
        self._load_models()

        # Optimizers
        self.policy_optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=config.critic_lr,
            weight_decay=config.weight_decay,
        )

        self.policy_scheduler = None
        self.critic_scheduler = None

        # Reward model (optional)
        self.reward_model = None
        if config.reward_model_name:
            self._load_reward_model()

    def _load_models(self) -> None:
        """Load policy, critic, and reference models."""
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer_name or self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Policy model
        self.policy = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.fsdp_config.param_dtype,
            attn_implementation="sdpa",
            trust_remote_code=self.config.trust_remote_code,
        )

        if self.fsdp_config.use_activation_checkpointing:
            self.policy.gradient_checkpointing_enable()

        self.policy = self.policy.to(self.device)

        # Get hidden size for value head
        model_config = AutoConfig.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
        )
        hidden_size = model_config.hidden_size

        # Critic model (separate or shared)
        if self.config.share_critic:
            # Add value head to policy
            self.value_head = ValueHead(hidden_size).to(self.device)
            self.critic = self.policy  # Share base model
        else:
            # Separate critic network
            self.critic = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=self.fsdp_config.param_dtype,
                attn_implementation="sdpa",
                trust_remote_code=self.config.trust_remote_code,
            )
            self.critic = self.critic.to(self.device)
            self.value_head = ValueHead(hidden_size).to(self.device)

        # Reference model (frozen)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.fsdp_config.param_dtype,
            attn_implementation="sdpa",
            trust_remote_code=self.config.trust_remote_code,
        )

        if self.fsdp_config.ref_model_offload:
            self.ref_model = self.ref_model.to("cpu")
        else:
            self.ref_model = self.ref_model.to(self.device)

        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def _load_reward_model(self) -> None:
        """Load reward model for scoring."""
        from dreadnode.core.training.ray.reward_model import RewardModel

        self.reward_model = RewardModel.from_pretrained(
            self.config.reward_model_name,
            torch_dtype=self.fsdp_config.param_dtype,
            trust_remote_code=self.config.trust_remote_code,
        )
        self.reward_model = self.reward_model.to(self.device)
        self.reward_model.eval()

    def train(
        self,
        prompts: list[str],
        reward_fn: Callable[[list[str], list[str]], list[float]] | None = None,
    ) -> dict[str, float]:
        """
        Run PPO training.

        Args:
            prompts: List of training prompts
            reward_fn: Optional reward function (prompts, completions) -> rewards.
                      Required if reward_model_name not set in config.

        Returns:
            Final training metrics
        """
        if reward_fn is None and self.reward_model is None:
            raise ValueError("Must provide either reward_fn or reward_model_name")

        # Create schedulers
        total_steps = self.config.max_steps
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        self._create_schedulers(total_steps, warmup_steps)

        # Training loop
        metrics = {}
        prompt_idx = 0

        while self.step < self.config.max_steps:
            # Sample batch of prompts
            batch_prompts = []
            for _ in range(self.config.batch_size):
                batch_prompts.append(prompts[prompt_idx % len(prompts)])
                prompt_idx += 1

            # Collect experience
            experience = self._collect_experience(batch_prompts, reward_fn)

            # PPO update
            ppo_metrics = self._ppo_update(experience)

            # Adaptive KL coefficient
            if self.config.kl_target is not None:
                self._update_kl_coef(ppo_metrics.get("kl_div", 0.0))

            # Logging
            if self.step % self.config.log_interval == 0:
                lr = self.policy_optimizer.param_groups[0]["lr"]
                print(
                    f"[Step {self.step}] "
                    f"reward: {ppo_metrics['mean_reward']:.4f} | "
                    f"policy_loss: {ppo_metrics['policy_loss']:.4f} | "
                    f"value_loss: {ppo_metrics['value_loss']:.4f} | "
                    f"kl: {ppo_metrics['kl_div']:.4f} | "
                    f"entropy: {ppo_metrics['entropy']:.4f} | "
                    f"kl_coef: {self.kl_coef:.4f} | "
                    f"lr: {lr:.2e}"
                )
                metrics = {"learning_rate": lr, **ppo_metrics}

            # Checkpointing
            if self.step > 0 and self.step % self.config.checkpoint_interval == 0:
                self.save_checkpoint()

            self.step += 1

        # Final checkpoint
        self.save_checkpoint()

        return metrics

    def _collect_experience(
        self,
        prompts: list[str],
        reward_fn: Callable[[list[str], list[str]], list[float]] | None,
    ) -> dict[str, Any]:
        """
        Collect experience by generating responses and computing rewards.

        Returns dict with:
            - prompt_ids: Tokenized prompts
            - response_ids: Generated response tokens
            - full_ids: prompt + response
            - attention_mask: Attention mask
            - old_logprobs: Log probs from current policy
            - ref_logprobs: Log probs from reference model
            - values: Value estimates
            - rewards: Reward for each response
            - advantages: GAE advantages
            - returns: Returns (advantages + values)
        """
        self.policy.eval()

        # Tokenize prompts
        prompt_encodings = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length - self.config.max_new_tokens,
            return_tensors="pt",
        )
        prompt_ids = prompt_encodings["input_ids"].to(self.device)
        prompt_mask = prompt_encodings["attention_mask"].to(self.device)

        # Generate responses
        with torch.no_grad():
            outputs = self.policy.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        full_ids = outputs.sequences
        response_ids = full_ids[:, prompt_ids.shape[1]:]

        # Create attention mask for full sequence
        attention_mask = torch.ones_like(full_ids)
        attention_mask[:, :prompt_ids.shape[1]] = prompt_mask

        # Decode responses for reward computation
        responses = self.tokenizer.batch_decode(
            response_ids, skip_special_tokens=True
        )

        # Compute rewards
        if reward_fn is not None:
            rewards = reward_fn(prompts, responses)
            rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        else:
            # Use reward model
            full_texts = [p + r for p, r in zip(prompts, responses)]
            with torch.no_grad():
                full_enc = self.tokenizer(
                    full_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    return_tensors="pt",
                )
                rewards = self.reward_model(
                    full_enc["input_ids"].to(self.device),
                    full_enc["attention_mask"].to(self.device),
                )

        # Compute log probs and values
        with torch.no_grad():
            # Policy log probs
            policy_outputs = self.policy(
                input_ids=full_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            old_logprobs = self._get_logprobs(
                policy_outputs.logits, full_ids, prompt_ids.shape[1]
            )

            # Reference log probs
            if self.fsdp_config.ref_model_offload:
                self.ref_model = self.ref_model.to(self.device)

            ref_outputs = self.ref_model(
                input_ids=full_ids,
                attention_mask=attention_mask,
            )
            ref_logprobs = self._get_logprobs(
                ref_outputs.logits, full_ids, prompt_ids.shape[1]
            )

            if self.fsdp_config.ref_model_offload:
                self.ref_model = self.ref_model.to("cpu")
                torch.cuda.empty_cache()

            # Value estimates
            if self.config.share_critic:
                hidden_states = policy_outputs.hidden_states[-1]
            else:
                critic_outputs = self.critic(
                    input_ids=full_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = critic_outputs.hidden_states[-1]

            values = self.value_head(hidden_states)
            # Only keep values for response tokens
            values = values[:, prompt_ids.shape[1] - 1:-1]

        # Compute GAE advantages
        advantages, returns = self._compute_gae(
            rewards=rewards,
            values=values,
            response_mask=(response_ids != self.tokenizer.pad_token_id).float(),
        )

        self.policy.train()

        return {
            "prompt_ids": prompt_ids,
            "response_ids": response_ids,
            "full_ids": full_ids,
            "attention_mask": attention_mask,
            "old_logprobs": old_logprobs,
            "ref_logprobs": ref_logprobs,
            "values": values,
            "rewards": rewards,
            "advantages": advantages,
            "returns": returns,
            "prompt_length": prompt_ids.shape[1],
        }

    def _get_logprobs(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        prompt_length: int,
    ) -> torch.Tensor:
        """Get log probabilities for response tokens only."""
        # Shift for next token prediction
        shift_logits = logits[:, prompt_length - 1:-1, :]
        shift_labels = labels[:, prompt_length:]

        log_probs = nn.functional.log_softmax(shift_logits, dim=-1)
        token_logprobs = log_probs.gather(
            dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        return token_logprobs

    def _compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        response_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE (Generalized Advantage Estimation).

        For text generation, reward is only at the end of the sequence.
        We distribute the reward across tokens weighted by position.

        Args:
            rewards: [batch_size] - final reward for each sequence
            values: [batch_size, seq_len] - value estimates
            response_mask: [batch_size, seq_len] - mask for response tokens

        Returns:
            advantages: [batch_size, seq_len]
            returns: [batch_size, seq_len]
        """
        batch_size, seq_len = values.shape
        device = values.device

        # For text generation, reward comes at the end
        # Create per-token rewards (0 except at last token)
        token_rewards = torch.zeros_like(values)
        response_lengths = response_mask.sum(dim=1).long()

        for i in range(batch_size):
            if response_lengths[i] > 0:
                # Put reward at last response token
                token_rewards[i, response_lengths[i] - 1] = rewards[i]

        # GAE computation (backwards through time)
        advantages = torch.zeros_like(values)
        last_gae = torch.zeros(batch_size, device=device)

        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = torch.zeros(batch_size, device=device)
            else:
                next_value = values[:, t + 1]

            delta = token_rewards[:, t] + self.config.gamma * next_value - values[:, t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * last_gae
            advantages[:, t] = last_gae

        # Mask out non-response tokens
        advantages = advantages * response_mask

        # Returns = advantages + values
        returns = advantages + values

        return advantages, returns

    def _ppo_update(self, experience: dict[str, Any]) -> dict[str, float]:
        """
        Perform PPO update on collected experience.

        Args:
            experience: Dict containing collected experience

        Returns:
            Training metrics
        """
        full_ids = experience["full_ids"]
        attention_mask = experience["attention_mask"]
        old_logprobs = experience["old_logprobs"]
        ref_logprobs = experience["ref_logprobs"]
        advantages = experience["advantages"]
        returns = experience["returns"]
        prompt_length = experience["prompt_length"]
        rewards = experience["rewards"]

        # Normalize advantages
        adv_mean = advantages[advantages != 0].mean()
        adv_std = advantages[advantages != 0].std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        # Response mask
        response_ids = experience["response_ids"]
        response_mask = (response_ids != self.tokenizer.pad_token_id).float()

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        n_updates = 0

        # PPO epochs
        for _ in range(self.config.ppo_epochs):
            # Mini-batch updates
            batch_size = full_ids.shape[0]
            indices = torch.randperm(batch_size)

            for start in range(0, batch_size, self.config.mini_batch_size):
                end = min(start + self.config.mini_batch_size, batch_size)
                mb_indices = indices[start:end]

                mb_full_ids = full_ids[mb_indices]
                mb_attention_mask = attention_mask[mb_indices]
                mb_old_logprobs = old_logprobs[mb_indices]
                mb_ref_logprobs = ref_logprobs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_response_mask = response_mask[mb_indices]

                # Forward pass
                policy_outputs = self.policy(
                    input_ids=mb_full_ids,
                    attention_mask=mb_attention_mask,
                    output_hidden_states=True,
                )

                new_logprobs = self._get_logprobs(
                    policy_outputs.logits, mb_full_ids, prompt_length
                )

                # Entropy
                shift_logits = policy_outputs.logits[:, prompt_length - 1:-1, :]
                probs = nn.functional.softmax(shift_logits, dim=-1)
                log_probs = nn.functional.log_softmax(shift_logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1)
                entropy = (entropy * mb_response_mask).sum() / mb_response_mask.sum()

                # Policy loss (PPO-Clip)
                ratio = torch.exp(new_logprobs - mb_old_logprobs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2)
                policy_loss = (policy_loss * mb_response_mask).sum() / mb_response_mask.sum()

                # KL penalty
                kl_div = mb_old_logprobs - new_logprobs
                kl_div = (kl_div * mb_response_mask).sum() / mb_response_mask.sum()

                # Value loss
                if self.config.share_critic:
                    hidden_states = policy_outputs.hidden_states[-1]
                else:
                    critic_outputs = self.critic(
                        input_ids=mb_full_ids,
                        attention_mask=mb_attention_mask,
                        output_hidden_states=True,
                    )
                    hidden_states = critic_outputs.hidden_states[-1]

                new_values = self.value_head(hidden_states)
                new_values = new_values[:, prompt_length - 1:-1]

                # Value clipping
                if self.config.value_clip_ratio > 0:
                    old_values = experience["values"][mb_indices]
                    clipped_values = old_values + torch.clamp(
                        new_values - old_values,
                        -self.config.value_clip_ratio,
                        self.config.value_clip_ratio,
                    )
                    vf_loss1 = (new_values - mb_returns) ** 2
                    vf_loss2 = (clipped_values - mb_returns) ** 2
                    value_loss = 0.5 * torch.max(vf_loss1, vf_loss2)
                else:
                    value_loss = 0.5 * (new_values - mb_returns) ** 2

                value_loss = (value_loss * mb_response_mask).sum() / mb_response_mask.sum()

                # Total loss
                loss = (
                    policy_loss
                    + self.kl_coef * kl_div
                    - self.config.entropy_coef * entropy
                    + value_loss
                )

                # Backward
                self.policy_optimizer.zero_grad()
                if not self.config.share_critic:
                    self.critic_optimizer.zero_grad()

                loss.backward()

                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.config.max_grad_norm
                    )
                    if not self.config.share_critic:
                        torch.nn.utils.clip_grad_norm_(
                            list(self.critic.parameters()) + list(self.value_head.parameters()),
                            self.config.max_grad_norm,
                        )

                self.policy_optimizer.step()
                if not self.config.share_critic:
                    self.critic_optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl += kl_div.item()
                n_updates += 1

        # Update schedulers
        if self.policy_scheduler is not None:
            self.policy_scheduler.step()
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "kl_div": total_kl / n_updates,
            "mean_reward": rewards.mean().item(),
            "std_reward": rewards.std().item(),
        }

    def _update_kl_coef(self, kl_div: float) -> None:
        """Adaptively update KL coefficient."""
        if self.config.kl_target is None:
            return

        if kl_div > 2.0 * self.config.kl_target:
            self.kl_coef *= 1.5
        elif kl_div < 0.5 * self.config.kl_target:
            self.kl_coef *= 0.5

        self.kl_coef = max(0.001, min(10.0, self.kl_coef))

    def _create_schedulers(self, total_steps: int, warmup_steps: int) -> None:
        """Create learning rate schedulers."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

        # Policy scheduler
        warmup = LinearLR(
            self.policy_optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        decay = CosineAnnealingLR(
            self.policy_optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=self.config.learning_rate * 0.1,
        )
        self.policy_scheduler = SequentialLR(
            self.policy_optimizer,
            schedulers=[warmup, decay],
            milestones=[warmup_steps],
        )

        # Critic scheduler (if separate)
        if not self.config.share_critic:
            warmup_c = LinearLR(
                self.critic_optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            decay_c = CosineAnnealingLR(
                self.critic_optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.config.critic_lr * 0.1,
            )
            self.critic_scheduler = SequentialLR(
                self.critic_optimizer,
                schedulers=[warmup_c, decay_c],
                milestones=[warmup_steps],
            )

    def save_checkpoint(self) -> None:
        """Save training checkpoint."""
        # Save to storage if available
        if self.storage is not None:
            local_model = self._save_to_storage()
            if local_model:
                print(f"Saved checkpoint to CAS: {local_model.name}")
                return

        # Fall back to file-based
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"ppo_step_{self.step}",
        )
        os.makedirs(checkpoint_path, exist_ok=True)

        self.policy.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)

        # Save value head
        torch.save(
            self.value_head.state_dict(),
            os.path.join(checkpoint_path, "value_head.pt"),
        )

        # Save training state
        torch.save(
            {
                "step": self.step,
                "kl_coef": self.kl_coef,
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            os.path.join(checkpoint_path, "training_state.pt"),
        )

        print(f"Saved checkpoint to {checkpoint_path}")

    def _save_to_storage(self) -> LocalModel | None:
        """Save checkpoint to CAS."""
        if self.storage is None:
            return None

        try:
            from dreadnode.models.local import LocalModel

            self._checkpoint_version += 1
            version = f"0.{self._checkpoint_version}.0"
            name = f"{self.checkpoint_name}-ppo-step{self.step}"

            return LocalModel.from_hf(
                model=self.policy,
                name=name,
                storage=self.storage,
                tokenizer=self.tokenizer,
                format="safetensors",
                task="text-generation",
                version=version,
            )
        except Exception as e:
            print(f"Failed to save to CAS: {e}")
            return None

    def get_policy(self) -> nn.Module:
        """Get the trained policy model."""
        return self.policy
