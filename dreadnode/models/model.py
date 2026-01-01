"""Model loader for published model packages."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from dreadnode.core.packaging.loader import BaseLoader
from dreadnode.core.packaging.manifest import ModelManifest

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


class Model(BaseLoader):
    """Loader for published model packages.

    This class loads models that have been published as DN packages
    with entry points in the 'dreadnode.models' group.

    Example:
        >>> from dreadnode.models import Model
        >>>
        >>> # Load a published model
        >>> model = Model("my-org-classifier")
        >>> print(model.framework)  # 'safetensors'
        >>> print(model.architecture)  # 'bert'
        >>>
        >>> # Load as HuggingFace model
        >>> hf_model = model.to_hf()
        >>> tokenizer = model.tokenizer()
    """

    entry_point_group = "dreadnode.models"
    manifest_class = ModelManifest

    @property
    def framework(self) -> str:
        """Model framework (safetensors, pytorch, onnx, etc.)."""
        return self.manifest.framework

    @property
    def task(self) -> str | None:
        """Model task type (classification, generation, etc.)."""
        return self.manifest.task

    @property
    def architecture(self) -> str | None:
        """Model architecture (bert, llama, vit, etc.)."""
        return self.manifest.architecture

    def model_path(self) -> Path:
        """Get the local path to the model directory.

        Resolves all model artifacts and returns the directory
        containing them.

        Returns:
            Path to local model directory.
        """
        # Resolve the first artifact to get the base path
        # Model artifacts are typically in a directory structure
        if not self.files:
            raise FileNotFoundError("No model artifacts in manifest")

        # Find the model config or weights file
        for artifact_path in self.files:
            if artifact_path.endswith("config.json") or artifact_path.endswith(".safetensors"):
                local_path = self.resolve(artifact_path)
                return local_path.parent

        # Fall back to resolving first file's parent
        local_path = self.resolve(self.files[0])
        return local_path.parent

    def to_hf(
        self,
        *,
        trust_remote_code: bool = False,
        torch_dtype: Any = None,
        device_map: str | None = None,
        **kwargs: Any,
    ) -> PreTrainedModel:
        """Load as HuggingFace PreTrainedModel.

        Args:
            trust_remote_code: Whether to trust remote code.
            torch_dtype: Torch dtype for model weights.
            device_map: Device map for model parallelism.
            **kwargs: Additional arguments for from_pretrained.

        Returns:
            HuggingFace PreTrainedModel.

        Example:
            >>> model = Model("my-classifier")
            >>> hf_model = model.to_hf(device_map="auto")
            >>> outputs = hf_model(**inputs)
        """
        from dreadnode.models.hf import load_model_from_pretrained, require_transformers

        require_transformers()

        # Resolve all artifacts first
        for artifact_path in self.files:
            self.resolve(artifact_path)

        model_dir = self.model_path()

        return load_model_from_pretrained(
            model_dir,
            task=self.task,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **kwargs,
        )

    def tokenizer(
        self,
        *,
        trust_remote_code: bool = False,
        **kwargs: Any,
    ) -> PreTrainedTokenizer:
        """Load the associated tokenizer.

        Args:
            trust_remote_code: Whether to trust remote code.
            **kwargs: Additional arguments for from_pretrained.

        Returns:
            HuggingFace PreTrainedTokenizer.

        Example:
            >>> model = Model("my-classifier")
            >>> tokenizer = model.tokenizer()
            >>> inputs = tokenizer("Hello world", return_tensors="pt")
        """
        from dreadnode.models.hf import load_tokenizer_from_pretrained, require_transformers

        require_transformers()

        # Resolve all artifacts first
        for artifact_path in self.files:
            self.resolve(artifact_path)

        model_dir = self.model_path()

        return load_tokenizer_from_pretrained(
            model_dir,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )


def load_model(name: str) -> Model:
    """Load a published model by name.

    Args:
        name: Model package name (entry point name).

    Returns:
        Model instance.

    Example:
        >>> model = load_model("my-classifier")
        >>> hf_model = model.to_hf()
    """
    return Model(name)
