"""
Configuration for Ray-based GRPO training.
"""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class VLLMConfig:
    """Configuration for vLLM inference."""

    tensor_parallel_size: int = 1
    """Number of GPUs for tensor parallelism per model replica."""

    gpu_memory_utilization: float = 0.85
    """Fraction of GPU memory to use for vLLM."""

    max_model_len: int | None = None
    """Maximum sequence length. If None, uses model's default."""

    enforce_eager: bool = False
    """Disable CUDA graphs for debugging."""

    dtype: str = "auto"
    """Data type for model weights (auto, float16, bfloat16)."""

    trust_remote_code: bool = True
    """Trust remote code in model repository."""

    enable_prefix_caching: bool = True
    """Enable prefix caching for KV cache reuse across same-prompt generations.

    This significantly speeds up GRPO where multiple completions share the same prompt.
    The KV cache for the prompt is computed once and reused for all completions.
    Enabled by default. Disable if experiencing issues with some models.
    """

    enable_chunked_prefill: bool = True
    """Enable chunked prefill for better memory efficiency.

    Works well with prefix caching for long prompts.
    """


@dataclass
class TrainingConfig:
    """Configuration for distributed training."""

    num_workers: int = 1
    """Number of training workers (GPUs)."""

    use_gpu: bool = True
    """Whether to use GPUs for training."""

    backend: Literal["deepspeed", "fsdp", "ddp"] = "deepspeed"
    """Distributed training backend."""

    deepspeed_stage: int = 2
    """DeepSpeed ZeRO stage (1, 2, or 3)."""

    gradient_checkpointing: bool = True
    """Enable gradient checkpointing to save memory."""

    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    """Mixed precision training mode."""


@dataclass
class GRPOLossConfig:
    """Configuration for GRPO loss function."""

    kl_coef: float = 0.1
    """KL divergence penalty coefficient."""

    clip_ratio: float = 0.2
    """PPO-style clipping ratio."""

    normalize_advantages: bool = True
    """Normalize advantages by std."""

    use_leave_one_out_baseline: bool = True
    """Use leave-one-out baseline for variance reduction."""

    token_level_loss: bool = False
    """Compute loss at token level vs sequence level."""


@dataclass
class RayGRPOConfig:
    """
    Complete configuration for Ray-based GRPO training.

    This configuration controls all aspects of GRPO training:
    - Model and tokenizer
    - Generation (vLLM)
    - Training (DeepSpeed/FSDP)
    - GRPO algorithm parameters
    """

    # Model
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    """Model name or path."""

    tokenizer_name: str | None = None
    """Tokenizer name (defaults to model_name)."""

    # GRPO Algorithm
    num_prompts_per_step: int = 8
    """Number of unique prompts per training step."""

    num_generations_per_prompt: int = 4
    """Number of completions to generate per prompt (G in GRPO)."""

    max_steps: int = 1000
    """Maximum training steps."""

    max_epochs: int = 10
    """Maximum training epochs."""

    # Generation
    max_new_tokens: int = 512
    """Maximum tokens to generate per completion."""

    temperature: float = 0.7
    """Sampling temperature."""

    top_p: float = 0.9
    """Top-p (nucleus) sampling."""

    # Training
    learning_rate: float = 1e-6
    """Learning rate."""

    weight_decay: float = 0.01
    """Weight decay."""

    warmup_ratio: float = 0.1
    """Warmup steps as fraction of total."""

    gradient_accumulation_steps: int = 1
    """Gradient accumulation steps."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm for clipping."""

    # Logging & Checkpointing
    log_interval: int = 10
    """Steps between logging."""

    eval_interval: int = 100
    """Steps between evaluation."""

    checkpoint_interval: int = 100
    """Steps between checkpoints."""

    checkpoint_dir: str = "./checkpoints"
    """Directory for checkpoints."""

    # Seed
    seed: int = 42
    """Random seed for reproducibility."""

    # Sub-configs
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    """vLLM inference configuration."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    """Distributed training configuration."""

    loss: GRPOLossConfig = field(default_factory=GRPOLossConfig)
    """GRPO loss configuration."""

    def __post_init__(self):
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name

    @property
    def train_batch_size(self) -> int:
        """Total batch size for training."""
        return self.num_prompts_per_step * self.num_generations_per_prompt

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "tokenizer_name": self.tokenizer_name,
            "num_prompts_per_step": self.num_prompts_per_step,
            "num_generations_per_prompt": self.num_generations_per_prompt,
            "max_steps": self.max_steps,
            "max_epochs": self.max_epochs,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "log_interval": self.log_interval,
            "eval_interval": self.eval_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "checkpoint_dir": self.checkpoint_dir,
            "seed": self.seed,
            "vllm": {
                "tensor_parallel_size": self.vllm.tensor_parallel_size,
                "gpu_memory_utilization": self.vllm.gpu_memory_utilization,
                "max_model_len": self.vllm.max_model_len,
                "enforce_eager": self.vllm.enforce_eager,
                "dtype": self.vllm.dtype,
                "trust_remote_code": self.vllm.trust_remote_code,
                "enable_prefix_caching": self.vllm.enable_prefix_caching,
                "enable_chunked_prefill": self.vllm.enable_chunked_prefill,
            },
            "training": {
                "num_workers": self.training.num_workers,
                "use_gpu": self.training.use_gpu,
                "backend": self.training.backend,
                "deepspeed_stage": self.training.deepspeed_stage,
                "gradient_checkpointing": self.training.gradient_checkpointing,
                "mixed_precision": self.training.mixed_precision,
            },
            "loss": {
                "kl_coef": self.loss.kl_coef,
                "clip_ratio": self.loss.clip_ratio,
                "normalize_advantages": self.loss.normalize_advantages,
                "use_leave_one_out_baseline": self.loss.use_leave_one_out_baseline,
                "token_level_loss": self.loss.token_level_loss,
            },
        }
