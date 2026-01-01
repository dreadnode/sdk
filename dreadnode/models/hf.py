"""HuggingFace transformers integration utilities for models."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


def require_transformers() -> None:
    """Raise ModuleNotFoundError if transformers is not installed."""
    if importlib.util.find_spec("transformers") is None:
        raise ModuleNotFoundError(
            "The 'transformers' package is required for HuggingFace model integration. "
            "Install it with: pip install dreadnode[training]"
        )


def load_model_from_pretrained(
    path: str | Path,
    *,
    task: str | None = None,
    trust_remote_code: bool = False,
    torch_dtype: Any = None,
    device_map: str | None = None,
    **kwargs: Any,
) -> PreTrainedModel:
    """Load a pretrained model from HuggingFace or local path.

    Args:
        path: Model path (HuggingFace Hub ID or local path).
        task: Task type to determine AutoModel class.
        trust_remote_code: Whether to trust remote code.
        torch_dtype: Torch dtype for model weights.
        device_map: Device map for model parallelism.
        **kwargs: Additional arguments for from_pretrained.

    Returns:
        Loaded pretrained model.
    """
    require_transformers()
    from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification

    # Map task to AutoModel class
    auto_class = AutoModel
    if task:
        task_lower = task.lower()
        if task_lower in ("causal-lm", "text-generation", "generation"):
            auto_class = AutoModelForCausalLM
        elif task_lower in ("sequence-classification", "classification"):
            auto_class = AutoModelForSequenceClassification
        # Add more task mappings as needed

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        **kwargs,
    }

    if torch_dtype is not None:
        load_kwargs["torch_dtype"] = torch_dtype
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    return auto_class.from_pretrained(str(path), **load_kwargs)


def load_tokenizer_from_pretrained(
    path: str | Path,
    *,
    trust_remote_code: bool = False,
    **kwargs: Any,
) -> PreTrainedTokenizer:
    """Load a pretrained tokenizer from HuggingFace or local path.

    Args:
        path: Tokenizer path (HuggingFace Hub ID or local path).
        trust_remote_code: Whether to trust remote code.
        **kwargs: Additional arguments for from_pretrained.

    Returns:
        Loaded pretrained tokenizer.
    """
    require_transformers()
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(path),
        trust_remote_code=trust_remote_code,
        **kwargs,
    )


def save_model_to_path(
    model: PreTrainedModel,
    path: Path,
    *,
    format: Literal["safetensors", "pytorch"] = "safetensors",
    **kwargs: Any,
) -> None:
    """Save a model to a local path.

    Args:
        model: Model to save.
        path: Destination path.
        format: Save format (safetensors or pytorch).
        **kwargs: Additional arguments for save_pretrained.
    """
    path.mkdir(parents=True, exist_ok=True)

    save_kwargs: dict[str, Any] = {**kwargs}
    if format == "safetensors":
        save_kwargs["safe_serialization"] = True
    else:
        save_kwargs["safe_serialization"] = False

    model.save_pretrained(str(path), **save_kwargs)


def save_tokenizer_to_path(
    tokenizer: PreTrainedTokenizer,
    path: Path,
    **kwargs: Any,
) -> None:
    """Save a tokenizer to a local path.

    Args:
        tokenizer: Tokenizer to save.
        path: Destination path.
        **kwargs: Additional arguments for save_pretrained.
    """
    path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(path), **kwargs)


def infer_model_info(model: PreTrainedModel) -> dict[str, Any]:
    """Extract model information for manifest.

    Args:
        model: Model to inspect.

    Returns:
        Dictionary with model metadata.
    """
    info: dict[str, Any] = {}

    if hasattr(model, "config"):
        config = model.config
        if hasattr(config, "model_type"):
            info["architecture"] = config.model_type
        if hasattr(config, "architectures") and config.architectures:
            info["architectures"] = config.architectures
        if hasattr(config, "num_parameters"):
            info["num_parameters"] = config.num_parameters
        elif hasattr(model, "num_parameters"):
            try:
                info["num_parameters"] = model.num_parameters()
            except Exception:
                pass

    return info
