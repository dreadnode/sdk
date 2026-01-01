"""Tests for HuggingFace model integration."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

transformers = pytest.importorskip("transformers")

from dreadnode.core.load import load
from dreadnode.core.storage.storage import Storage
from dreadnode.models.hf import (
    infer_model_info,
    require_transformers,
)
from dreadnode.models.local import LocalModel


@pytest.fixture
def temp_storage(tmp_path):
    """Create a temporary storage instance."""
    cache_dir = tmp_path / ".dreadnode"
    cache_dir.mkdir()
    return Storage(cache=cache_dir)


@pytest.fixture
def mock_model():
    """Create a mock HuggingFace model."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.model_type = "bert"
    model.config.architectures = ["BertForSequenceClassification"]
    model.num_parameters.return_value = 110_000_000
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock HuggingFace tokenizer."""
    tokenizer = MagicMock()
    return tokenizer


# ==============================================================================
# HF Utility Tests
# ==============================================================================


class TestRequireTransformers:
    """Tests for require_transformers function."""

    def test_require_transformers_succeeds(self):
        """Test that require_transformers doesn't raise when installed."""
        require_transformers()

    def test_require_transformers_raises_when_missing(self):
        """Test that require_transformers raises when not installed."""
        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(ModuleNotFoundError, match="transformers"):
                require_transformers()


class TestInferModelInfo:
    """Tests for model info inference."""

    def test_infer_model_info_basic(self, mock_model):
        """Test extracting model info."""
        # Configure model.config to not have num_parameters
        mock_model.config = MagicMock(spec=["model_type", "architectures"])
        mock_model.config.model_type = "bert"
        mock_model.config.architectures = ["BertForSequenceClassification"]

        info = infer_model_info(mock_model)

        assert info["architecture"] == "bert"
        assert info["architectures"] == ["BertForSequenceClassification"]
        assert info["num_parameters"] == 110_000_000

    def test_infer_model_info_minimal(self):
        """Test with minimal model config."""
        model = MagicMock()
        model.config = MagicMock(spec=[])  # Empty spec

        info = infer_model_info(model)
        # Should not raise, just return empty or partial info
        assert isinstance(info, dict)


# ==============================================================================
# LocalModel Tests
# ==============================================================================


class TestLocalModelFromHF:
    """Tests for LocalModel.from_hf()."""

    def test_from_hf_creates_manifest(self, mock_model, mock_tokenizer, temp_storage):
        """Test that from_hf creates proper manifest."""
        with patch.object(mock_model, "save_pretrained") as save_model:
            with patch.object(mock_tokenizer, "save_pretrained") as save_tokenizer:
                # Mock save_pretrained to create files
                def create_model_files(path, **kwargs):
                    Path(path).mkdir(parents=True, exist_ok=True)
                    (Path(path) / "config.json").write_text('{"model_type": "bert"}')
                    (Path(path) / "model.safetensors").write_bytes(b"mock weights")

                def create_tokenizer_files(path, **kwargs):
                    Path(path).mkdir(parents=True, exist_ok=True)
                    (Path(path) / "tokenizer.json").write_text("{}")

                save_model.side_effect = create_model_files
                save_tokenizer.side_effect = create_tokenizer_files

                local_model = LocalModel.from_hf(
                    mock_model,
                    "test-model",
                    temp_storage,
                    tokenizer=mock_tokenizer,
                )

        assert local_model.name == "test-model"
        assert local_model.version == "0.1.0"
        assert local_model.framework == "safetensors"
        assert len(local_model.files) > 0

    def test_from_hf_custom_version(self, mock_model, temp_storage):
        """Test from_hf with custom version."""
        with patch.object(mock_model, "save_pretrained") as save_model:

            def create_files(path, **kwargs):
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "config.json").write_text("{}")
                (Path(path) / "model.safetensors").write_bytes(b"data")

            save_model.side_effect = create_files

            local_model = LocalModel.from_hf(
                mock_model,
                "test-model",
                temp_storage,
                version="1.0.0",
            )

        assert local_model.version == "1.0.0"

    def test_from_hf_pytorch_format(self, mock_model, temp_storage):
        """Test from_hf with pytorch format."""
        with patch.object(mock_model, "save_pretrained") as save_model:

            def create_files(path, **kwargs):
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "config.json").write_text("{}")
                (Path(path) / "pytorch_model.bin").write_bytes(b"data")

            save_model.side_effect = create_files

            local_model = LocalModel.from_hf(
                mock_model,
                "test-model",
                temp_storage,
                format="pytorch",
            )

        assert local_model.framework == "pytorch"


class TestLocalModelReload:
    """Tests for reloading existing LocalModel."""

    def test_reload_existing(self, mock_model, temp_storage):
        """Test loading an existing LocalModel by name."""
        with patch.object(mock_model, "save_pretrained") as save_model:

            def create_files(path, **kwargs):
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "config.json").write_text("{}")
                (Path(path) / "model.safetensors").write_bytes(b"data")

            save_model.side_effect = create_files

            LocalModel.from_hf(mock_model, "test", temp_storage)

        # Reload
        local_model = LocalModel("test", temp_storage)

        assert local_model.name == "test"
        assert local_model.framework == "safetensors"

    def test_reload_nonexistent_raises(self, temp_storage):
        """Test that loading nonexistent model raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            LocalModel("nonexistent", temp_storage)


class TestLocalModelMethods:
    """Tests for LocalModel methods."""

    def test_model_path_reconstructs_directory(self, mock_model, temp_storage):
        """Test that model_path reconstructs the model directory."""
        with patch.object(mock_model, "save_pretrained") as save_model:

            def create_files(path, **kwargs):
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "config.json").write_text('{"test": true}')
                (Path(path) / "model.safetensors").write_bytes(b"weights")

            save_model.side_effect = create_files

            local_model = LocalModel.from_hf(mock_model, "test", temp_storage)

        model_dir = local_model.model_path()

        assert model_dir.exists()
        assert (model_dir / "config.json").exists()
        assert (model_dir / "model.safetensors").exists()

    def test_publish_not_implemented(self, mock_model, temp_storage):
        """Test that publish() raises NotImplementedError."""
        with patch.object(mock_model, "save_pretrained") as save_model:

            def create_files(path, **kwargs):
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "config.json").write_text("{}")

            save_model.side_effect = create_files

            local_model = LocalModel.from_hf(mock_model, "test", temp_storage)

        with pytest.raises(NotImplementedError):
            local_model.publish()

    def test_repr(self, mock_model, temp_storage):
        """Test __repr__ method."""
        with patch.object(mock_model, "save_pretrained") as save_model:

            def create_files(path, **kwargs):
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "config.json").write_text("{}")

            save_model.side_effect = create_files

            local_model = LocalModel.from_hf(mock_model, "test", temp_storage)

        repr_str = repr(local_model)
        assert "LocalModel" in repr_str
        assert "test" in repr_str
        assert "safetensors" in repr_str


# ==============================================================================
# Unified Load Tests
# ==============================================================================


class TestUnifiedLoadModel:
    """Tests for the unified dn.load() function with models."""

    def test_load_hf_model_scheme(self, temp_storage):
        """Test loading with hf-model:// scheme."""
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.model_type = "bert"

        def create_files(path, **kwargs):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "config.json").write_text("{}")
            (Path(path) / "model.safetensors").write_bytes(b"data")

        mock_model.save_pretrained = MagicMock(side_effect=create_files)

        with patch("dreadnode.models.hf.load_model_from_pretrained", return_value=mock_model):
            with patch(
                "dreadnode.models.hf.load_tokenizer_from_pretrained",
                side_effect=Exception("No tokenizer"),
            ):
                local_model = load("hf-model://bert-base-uncased", storage=temp_storage)

        assert isinstance(local_model, LocalModel)
        assert local_model.name == "bert-base-uncased"


# ==============================================================================
# Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_manifest_fields(self, mock_model, temp_storage):
        """Test that manifest has all expected fields."""
        with patch.object(mock_model, "save_pretrained") as save_model:

            def create_files(path, **kwargs):
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "config.json").write_text("{}")

            save_model.side_effect = create_files

            local_model = LocalModel.from_hf(
                mock_model,
                "test",
                temp_storage,
                task="classification",
            )

        assert local_model.task == "classification"
        assert local_model.architecture == "bert"
