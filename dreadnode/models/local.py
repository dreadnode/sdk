"""Local model storage without package installation."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from dreadnode.core.packaging.manifest import ModelManifest
from dreadnode.core.storage.storage import Storage, hash_file
from dreadnode.core.settings import DEFAULT_CACHE_DIR

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


class LocalModel:
    """Model stored in CAS, usable without package installation.

    This class provides a way to work with models stored in the
    Content-Addressable Storage without requiring them to be installed
    as Python packages with entry points.

    Example:
        >>> from dreadnode.models import LocalModel
        >>> from dreadnode.core.storage import Storage
        >>>
        >>> storage = Storage()
        >>>
        >>> # Save a HuggingFace model to CAS
        >>> from transformers import AutoModelForSequenceClassification
        >>> hf_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
        >>> local_model = LocalModel.from_hf(hf_model, "my-bert", storage)
        >>>
        >>> # Load and use
        >>> model = local_model.to_hf()
        >>> tokenizer = local_model.tokenizer()
    """

    def __init__(
        self,
        name: str,
        storage: Storage,
        version: str | None = None,
    ):
        """Load a local model by name.

        Args:
            name: Model name.
            storage: Storage instance for CAS access.
            version: Specific version to load. If None, loads latest.
        """
        self.name = name
        self.storage = storage

        if version is None:
            version = storage.latest_version("models", name)
            if version is None:
                raise FileNotFoundError(f"Model not found: {name}")

        self.version = version
        self._manifest: ModelManifest | None = None
        self._model_dir: Path | None = None

    @property
    def manifest(self) -> ModelManifest:
        """Load and cache the manifest."""
        if self._manifest is None:
            content = self.storage.get_manifest("models", self.name, self.version)
            self._manifest = ModelManifest.model_validate_json(content)
        return self._manifest

    @property
    def framework(self) -> str:
        """Model framework (safetensors, pytorch, onnx, etc.)."""
        return self.manifest.framework

    @property
    def task(self) -> str | None:
        """Model task type."""
        return self.manifest.task

    @property
    def architecture(self) -> str | None:
        """Model architecture."""
        return self.manifest.architecture

    @property
    def files(self) -> list[str]:
        """List of artifact file paths."""
        return list(self.manifest.artifacts.keys())

    def _resolve(self, path: str) -> Path:
        """Resolve artifact path to local file."""
        if path not in self.manifest.artifacts:
            raise FileNotFoundError(f"Artifact not in manifest: {path}")

        oid = self.manifest.artifacts[path]
        local_path = self.storage.blob_path(oid)

        if not local_path.exists():
            self.storage.download_blob(oid)

        return local_path

    def model_path(self) -> Path:
        """Get the local path to the model directory.

        Reconstructs the model directory structure from CAS blobs.

        Returns:
            Path to local model directory.
        """
        if self._model_dir is not None and self._model_dir.exists():
            return self._model_dir

        # Create a temp directory to reconstruct model structure
        model_dir = Path(tempfile.mkdtemp(prefix=f"dn_model_{self.name}_"))

        for artifact_path in self.files:
            blob_path = self._resolve(artifact_path)
            dest_path = model_dir / artifact_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            # Copy or link the blob
            shutil.copy2(blob_path, dest_path)

        self._model_dir = model_dir
        return model_dir

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
        """
        from dreadnode.models.hf import load_model_from_pretrained, require_transformers

        require_transformers()

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
        """
        from dreadnode.models.hf import load_tokenizer_from_pretrained, require_transformers

        require_transformers()

        model_dir = self.model_path()

        return load_tokenizer_from_pretrained(
            model_dir,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

    @classmethod
    def from_hf(
        cls,
        model: PreTrainedModel,
        name: str,
        storage: Storage,
        *,
        tokenizer: PreTrainedTokenizer | None = None,
        format: Literal["safetensors", "pytorch"] = "safetensors",
        task: str | None = None,
        version: str = "0.1.0",
    ) -> LocalModel:
        """Store a HuggingFace model in CAS and return LocalModel.

        Args:
            model: HuggingFace PreTrainedModel to store.
            name: Name for the model.
            storage: Storage instance for CAS access.
            tokenizer: Optional tokenizer to save alongside model.
            format: Save format (safetensors or pytorch).
            task: Task type for manifest.
            version: Version string.

        Returns:
            LocalModel instance for the stored model.

        Example:
            >>> from transformers import AutoModelForCausalLM, AutoTokenizer
            >>> model = AutoModelForCausalLM.from_pretrained("gpt2")
            >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
            >>> local = LocalModel.from_hf(model, "my-gpt2", storage, tokenizer=tokenizer)
        """
        from dreadnode.models.hf import (
            infer_model_info,
            require_transformers,
            save_model_to_path,
            save_tokenizer_to_path,
        )

        require_transformers()

        artifacts: dict[str, str] = {}

        # Save to temp directory first
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Save model
            save_model_to_path(model, tmp_path, format=format)

            # Save tokenizer if provided
            if tokenizer is not None:
                save_tokenizer_to_path(tokenizer, tmp_path)

            # Hash and store each file in CAS
            for file_path in tmp_path.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(tmp_path)
                    file_hash = hash_file(file_path)
                    oid = f"sha256:{file_hash}"
                    storage.store_blob(oid, file_path)
                    artifacts[str(rel_path)] = oid

        # Infer model info
        model_info = infer_model_info(model)
        architecture = model_info.get("architecture")

        # Create manifest
        manifest = ModelManifest(
            framework=format,
            task=task,
            architecture=architecture,
            artifacts=artifacts,
        )

        # Store manifest
        manifest_json = manifest.model_dump_json(indent=2)
        storage.store_manifest("models", name, version, manifest_json)

        return cls(name, storage, version)

    def publish(self, version: str | None = None) -> None:
        """Create a DN package for signing and distribution.

        Args:
            version: Version for the package. If None, uses current version.

        Raises:
            NotImplementedError: Package creation not yet implemented.
        """
        raise NotImplementedError(
            "Package publishing is not yet implemented. Use the CLI to create and publish packages."
        )

    def __repr__(self) -> str:
        return f"LocalModel(name={self.name!r}, version={self.version!r}, framework={self.framework!r})"


def load_model(
    path: str,
    *,
    name: str | None = None,
    storage: Storage | None = None,
    task: str | None = None,
    format: Literal["safetensors", "pytorch"] = "safetensors",
    version: str = "0.1.0",
    **kwargs: Any,
) -> LocalModel:
    """Load a model from HuggingFace Hub and store in local CAS.

    This is a convenience function that downloads a model from
    HuggingFace Hub and stores it in Content-Addressable Storage.

    Args:
        path: HuggingFace model path (e.g., "bert-base-uncased").
        name: Name to store the model as. Defaults to the path.
        storage: Storage instance. If None, creates default storage.
        task: Task type for the model.
        format: Storage format (safetensors or pytorch).
        version: Version string for the stored model.
        **kwargs: Additional arguments passed to from_pretrained.

    Returns:
        LocalModel instance with the loaded model.

    Example:
        >>> from dreadnode.models import load_model
        >>>
        >>> # Load and store a HuggingFace model
        >>> model = load_model("bert-base-uncased", task="classification")
        >>> hf_model = model.to_hf()
    """
    from dreadnode.models.hf import (
        load_model_from_pretrained,
        load_tokenizer_from_pretrained,
        require_transformers,
    )

    require_transformers()

    # Create default storage if not provided
    if storage is None:
        storage = Storage(cache=DEFAULT_CACHE_DIR)

    # Use path as name if not specified
    if name is None:
        name = path.replace("/", "-").replace(":", "-")

    # Load from HuggingFace
    hf_model = load_model_from_pretrained(path, task=task, **kwargs)

    # Try to load tokenizer (may not exist for all models)
    try:
        hf_tokenizer = load_tokenizer_from_pretrained(path, **kwargs)
    except Exception:
        hf_tokenizer = None

    # Store in CAS and return LocalModel
    return LocalModel.from_hf(
        hf_model,
        name=name,
        storage=storage,
        tokenizer=hf_tokenizer,
        format=format,
        task=task,
        version=version,
    )
