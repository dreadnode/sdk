"""Tests for HuggingFace datasets integration."""

from unittest.mock import MagicMock, patch

import pytest

# Skip all tests if datasets not installed
datasets = pytest.importorskip("datasets")
import pyarrow as pa

from dreadnode.core.load import load
from dreadnode.core.storage.storage import Storage
from dreadnode.datasets.hf import (
    arrow_to_hf,
    hf_to_arrow,
    infer_schema,
    require_datasets,
)
from dreadnode.datasets.local import LocalDataset, load_dataset, load_file, write_table

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def sample_table():
    """Create a sample PyArrow table."""
    return pa.table(
        {
            "id": [1, 2, 3],
            "text": ["hello", "world", "test"],
            "score": [0.1, 0.5, 0.9],
        }
    )


@pytest.fixture
def sample_hf_dataset():
    """Create a sample HuggingFace dataset."""
    return datasets.Dataset.from_dict(
        {
            "id": [1, 2, 3],
            "text": ["hello", "world", "test"],
            "score": [0.1, 0.5, 0.9],
        }
    )


@pytest.fixture
def sample_hf_dataset_dict():
    """Create a sample HuggingFace DatasetDict with splits."""
    return datasets.DatasetDict(
        {
            "train": datasets.Dataset.from_dict(
                {
                    "id": [1, 2],
                    "text": ["hello", "world"],
                }
            ),
            "test": datasets.Dataset.from_dict(
                {
                    "id": [3],
                    "text": ["test"],
                }
            ),
        }
    )


@pytest.fixture
def temp_storage(tmp_path):
    """Create a temporary storage instance."""
    cache_dir = tmp_path / ".dreadnode"
    cache_dir.mkdir()
    return Storage(cache=cache_dir)


# ==============================================================================
# HF Conversion Tests
# ==============================================================================


class TestArrowConversion:
    """Tests for PyArrow <-> HF Dataset conversion."""

    def test_arrow_to_hf_basic(self, sample_table):
        """Test basic Arrow to HF conversion."""
        hf_ds = arrow_to_hf(sample_table)

        assert len(hf_ds) == 3
        assert hf_ds.column_names == ["id", "text", "score"]
        assert hf_ds["id"] == [1, 2, 3]
        assert hf_ds["text"] == ["hello", "world", "test"]

    def test_hf_to_arrow_basic(self, sample_hf_dataset):
        """Test HF to Arrow conversion."""
        table = hf_to_arrow(sample_hf_dataset)

        assert isinstance(table, pa.Table)
        assert table.num_rows == 3
        assert table.column_names == ["id", "text", "score"]

    def test_roundtrip(self, sample_hf_dataset):
        """Test HF -> Arrow -> HF roundtrip."""
        table = hf_to_arrow(sample_hf_dataset)
        hf_ds2 = arrow_to_hf(table)

        assert len(hf_ds2) == len(sample_hf_dataset)
        assert hf_ds2.column_names == sample_hf_dataset.column_names
        assert hf_ds2["id"] == sample_hf_dataset["id"]

    def test_infer_schema(self, sample_hf_dataset):
        """Test schema inference."""
        schema = infer_schema(sample_hf_dataset)

        assert "id" in schema
        assert "text" in schema
        assert "score" in schema


class TestRequireDatasets:
    """Tests for require_datasets function."""

    def test_require_datasets_succeeds(self):
        """Test that require_datasets doesn't raise when datasets is installed."""
        # Should not raise
        require_datasets()

    def test_require_datasets_raises_when_missing(self):
        """Test that require_datasets raises when datasets is not installed."""
        with patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(ModuleNotFoundError, match="datasets"):
                require_datasets()


# ==============================================================================
# File I/O Tests
# ==============================================================================


class TestFileIO:
    """Tests for file loading and writing."""

    def test_write_and_load_parquet(self, sample_table, tmp_path):
        """Test writing and loading parquet format."""
        path = tmp_path / "data.parquet"
        write_table(sample_table, path, "parquet")

        loaded = load_file(path, "parquet")
        assert loaded.num_rows == 3
        assert loaded.column_names == sample_table.column_names

    def test_write_and_load_arrow(self, sample_table, tmp_path):
        """Test writing and loading arrow/feather format."""
        path = tmp_path / "data.arrow"
        write_table(sample_table, path, "arrow")

        loaded = load_file(path, "arrow")
        assert loaded.num_rows == 3

    def test_load_infers_format(self, sample_table, tmp_path):
        """Test that load_file infers format from extension."""
        path = tmp_path / "data.parquet"
        write_table(sample_table, path, "parquet")

        # Load without specifying format
        loaded = load_file(path)
        assert loaded.num_rows == 3

    def test_load_unknown_format_raises(self, tmp_path):
        """Test that unknown format raises ValueError."""
        path = tmp_path / "data.xyz"
        path.write_text("dummy")

        with pytest.raises(ValueError, match="Unknown format"):
            load_file(path, "xyz")


# ==============================================================================
# LocalDataset Tests
# ==============================================================================


class TestLocalDatasetFromHF:
    """Tests for LocalDataset.from_hf()."""

    def test_from_hf_single_dataset(self, sample_hf_dataset, temp_storage):
        """Test creating LocalDataset from single HF Dataset."""
        local_ds = LocalDataset.from_hf(
            sample_hf_dataset,
            "test-dataset",
            temp_storage,
        )

        assert local_ds.name == "test-dataset"
        assert local_ds.version == "0.1.0"
        assert local_ds.format == "parquet"
        assert local_ds.row_count == 3
        assert len(local_ds.files) == 1

    def test_from_hf_dataset_dict(self, sample_hf_dataset_dict, temp_storage):
        """Test creating LocalDataset from DatasetDict with splits."""
        local_ds = LocalDataset.from_hf(
            sample_hf_dataset_dict,
            "test-splits",
            temp_storage,
        )

        assert local_ds.name == "test-splits"
        assert local_ds.row_count == 3  # 2 + 1
        assert local_ds.splits == ["train", "test"]
        assert len(local_ds.files) == 2

    def test_from_hf_custom_format(self, sample_hf_dataset, temp_storage):
        """Test creating LocalDataset with custom format."""
        local_ds = LocalDataset.from_hf(
            sample_hf_dataset,
            "test-arrow",
            temp_storage,
            format="arrow",
        )

        assert local_ds.format == "arrow"

    def test_from_hf_custom_version(self, sample_hf_dataset, temp_storage):
        """Test creating LocalDataset with custom version."""
        local_ds = LocalDataset.from_hf(
            sample_hf_dataset,
            "test-version",
            temp_storage,
            version="1.0.0",
        )

        assert local_ds.version == "1.0.0"


class TestLocalDatasetLoad:
    """Tests for LocalDataset loading methods."""

    def test_load_returns_table(self, sample_hf_dataset, temp_storage):
        """Test that load() returns PyArrow Table."""
        local_ds = LocalDataset.from_hf(sample_hf_dataset, "test", temp_storage)

        table = local_ds.load()

        assert isinstance(table, pa.Table)
        assert table.num_rows == 3

    def test_to_hf_returns_dataset(self, sample_hf_dataset, temp_storage):
        """Test that to_hf() returns HF Dataset."""
        local_ds = LocalDataset.from_hf(sample_hf_dataset, "test", temp_storage)

        hf_ds = local_ds.to_hf()

        assert isinstance(hf_ds, datasets.Dataset)
        assert len(hf_ds) == 3

    def test_to_pandas_returns_dataframe(self, sample_hf_dataset, temp_storage):
        """Test that to_pandas() returns DataFrame."""
        local_ds = LocalDataset.from_hf(sample_hf_dataset, "test", temp_storage)

        df = local_ds.to_pandas()

        assert len(df) == 3
        assert list(df.columns) == ["id", "text", "score"]

    def test_load_with_split(self, sample_hf_dataset_dict, temp_storage):
        """Test loading specific split."""
        local_ds = LocalDataset.from_hf(
            sample_hf_dataset_dict,
            "test",
            temp_storage,
        )

        train_table = local_ds.load("train")
        test_table = local_ds.load("test")

        assert train_table.num_rows == 2
        assert test_table.num_rows == 1

    def test_load_unknown_split_raises(self, sample_hf_dataset_dict, temp_storage):
        """Test that loading unknown split raises ValueError."""
        local_ds = LocalDataset.from_hf(
            sample_hf_dataset_dict,
            "test",
            temp_storage,
        )

        with pytest.raises(ValueError, match="Unknown split"):
            local_ds.load("validation")


class TestLocalDatasetReload:
    """Tests for reloading existing LocalDataset."""

    def test_reload_existing(self, sample_hf_dataset, temp_storage):
        """Test loading an existing LocalDataset by name."""
        # Create first
        LocalDataset.from_hf(sample_hf_dataset, "test", temp_storage)

        # Reload
        local_ds = LocalDataset("test", temp_storage)

        assert local_ds.name == "test"
        assert local_ds.row_count == 3

    def test_reload_specific_version(self, sample_hf_dataset, temp_storage):
        """Test loading specific version."""
        LocalDataset.from_hf(
            sample_hf_dataset,
            "test",
            temp_storage,
            version="1.0.0",
        )

        local_ds = LocalDataset("test", temp_storage, version="1.0.0")

        assert local_ds.version == "1.0.0"

    def test_reload_nonexistent_raises(self, temp_storage):
        """Test that loading nonexistent dataset raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            LocalDataset("nonexistent", temp_storage)


# ==============================================================================
# HF Operations Tests
# ==============================================================================


class TestHFOperations:
    """Tests for HuggingFace operations on converted datasets."""

    def test_map_operation(self, sample_hf_dataset, temp_storage):
        """Test that .map() works on converted dataset."""
        local_ds = LocalDataset.from_hf(sample_hf_dataset, "test", temp_storage)
        hf_ds = local_ds.to_hf()

        mapped = hf_ds.map(lambda x: {"upper": x["text"].upper()})

        assert "upper" in mapped.column_names
        assert mapped["upper"] == ["HELLO", "WORLD", "TEST"]

    def test_filter_operation(self, sample_hf_dataset, temp_storage):
        """Test that .filter() works."""
        local_ds = LocalDataset.from_hf(sample_hf_dataset, "test", temp_storage)
        hf_ds = local_ds.to_hf()

        filtered = hf_ds.filter(lambda x: x["id"] > 1)

        assert len(filtered) == 2

    def test_shuffle_operation(self, temp_storage):
        """Test that .shuffle() works."""
        large_ds = datasets.Dataset.from_dict({"x": list(range(100))})
        local_ds = LocalDataset.from_hf(large_ds, "test", temp_storage)
        hf_ds = local_ds.to_hf()

        shuffled = hf_ds.shuffle(seed=42)

        assert len(shuffled) == 100
        assert shuffled["x"] != list(range(100))

    def test_select_operation(self, sample_hf_dataset, temp_storage):
        """Test that .select() works."""
        local_ds = LocalDataset.from_hf(sample_hf_dataset, "test", temp_storage)
        hf_ds = local_ds.to_hf()

        selected = hf_ds.select([0, 2])

        assert len(selected) == 2
        assert selected["id"] == [1, 3]

    def test_train_test_split(self, temp_storage):
        """Test .train_test_split()."""
        large_ds = datasets.Dataset.from_dict({"x": list(range(100))})
        local_ds = LocalDataset.from_hf(large_ds, "test", temp_storage)
        hf_ds = local_ds.to_hf()

        splits = hf_ds.train_test_split(test_size=0.2, seed=42)

        assert len(splits["train"]) == 80
        assert len(splits["test"]) == 20


class TestHFTorchFormat:
    """Tests for torch format integration."""

    def test_set_format_torch(self, temp_storage):
        """Test .set_format('torch') for DataLoader."""
        torch = pytest.importorskip("torch")

        hf_ds = datasets.Dataset.from_dict({"x": [1.0, 2.0, 3.0]})
        local_ds = LocalDataset.from_hf(hf_ds, "test", temp_storage)
        loaded = local_ds.to_hf()

        loaded.set_format("torch")

        assert isinstance(loaded[0]["x"], torch.Tensor)


# ==============================================================================
# Roundtrip Tests
# ==============================================================================


class TestRoundTrip:
    """Tests for complete roundtrip conversions."""

    def test_hf_to_local_to_hf(self, sample_hf_dataset, temp_storage):
        """Test HF -> LocalDataset -> HF roundtrip."""
        local_ds = LocalDataset.from_hf(sample_hf_dataset, "test", temp_storage)
        hf_ds = local_ds.to_hf()

        assert len(hf_ds) == len(sample_hf_dataset)
        assert hf_ds.column_names == sample_hf_dataset.column_names
        assert hf_ds["id"] == sample_hf_dataset["id"]
        assert hf_ds["text"] == sample_hf_dataset["text"]

    def test_preserves_types(self, temp_storage):
        """Test that data types are preserved in roundtrip."""
        original = datasets.Dataset.from_dict(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
                "bool_col": [True, False, True],
            }
        )

        local_ds = LocalDataset.from_hf(original, "test", temp_storage)
        loaded = local_ds.to_hf()

        # Check values match
        assert loaded["int_col"] == original["int_col"]
        assert loaded["float_col"] == original["float_col"]
        assert loaded["str_col"] == original["str_col"]
        assert loaded["bool_col"] == original["bool_col"]


# ==============================================================================
# Edge Cases
# ==============================================================================


class TestUnifiedLoad:
    """Tests for the unified dn.load() function."""

    def test_load_hf_scheme(self, temp_storage):
        """Test loading with hf:// scheme."""
        from unittest.mock import patch

        mock_ds = datasets.Dataset.from_dict({"x": [1, 2, 3]})

        with patch("datasets.load_dataset", return_value=mock_ds):
            local_ds = load("hf://squad", storage=temp_storage)

        assert isinstance(local_ds, LocalDataset)
        assert local_ds.name == "squad"

    def test_load_plain_path_defaults_to_hf(self, temp_storage):
        """Test that plain path without scheme defaults to HuggingFace."""
        from unittest.mock import patch

        mock_ds = datasets.Dataset.from_dict({"x": [1, 2, 3]})

        with patch("datasets.load_dataset", return_value=mock_ds):
            local_ds = load("imdb", storage=temp_storage)

        assert isinstance(local_ds, LocalDataset)
        assert local_ds.name == "imdb"

    def test_load_hf_with_name_override(self, temp_storage):
        """Test loading HF dataset with custom name."""
        from unittest.mock import patch

        mock_ds = datasets.Dataset.from_dict({"x": [1, 2, 3]})

        with patch("datasets.load_dataset", return_value=mock_ds):
            local_ds = load("hf://squad", name="my-squad", storage=temp_storage)

        assert local_ds.name == "my-squad"

    def test_load_unknown_scheme_raises(self, temp_storage):
        """Test that unknown scheme raises ValueError."""
        with pytest.raises(ValueError, match="Unknown URI scheme"):
            load("unknown://something", storage=temp_storage)


class TestLoadDatasetFunction:
    """Tests for the load_dataset convenience function."""

    def test_load_dataset_from_dict(self, temp_storage):
        """Test load_dataset with a simple in-memory dataset."""
        # Create a mock HF dataset and use from_hf directly
        # (We can't test actual HF Hub loading without network)
        hf_ds = datasets.Dataset.from_dict({"x": [1, 2, 3]})
        local_ds = LocalDataset.from_hf(hf_ds, "test", temp_storage)

        assert local_ds.name == "test"
        assert local_ds.row_count == 3

    def test_load_dataset_default_name(self, temp_storage):
        """Test that load_dataset uses path as default name."""
        # Mock the HF load_dataset to avoid network calls
        from unittest.mock import patch

        mock_ds = datasets.Dataset.from_dict({"text": ["hello"]})

        with patch("datasets.load_dataset", return_value=mock_ds):
            local_ds = load_dataset("squad", storage=temp_storage)

        assert local_ds.name == "squad"

    def test_load_dataset_custom_name(self, temp_storage):
        """Test load_dataset with custom name."""
        from unittest.mock import patch

        mock_ds = datasets.Dataset.from_dict({"text": ["hello"]})

        with patch("datasets.load_dataset", return_value=mock_ds):
            local_ds = load_dataset("squad", name="my-squad", storage=temp_storage)

        assert local_ds.name == "my-squad"

    def test_load_dataset_passes_kwargs(self, temp_storage):
        """Test that kwargs are passed to HF load_dataset."""
        from unittest.mock import patch

        mock_ds = datasets.Dataset.from_dict({"text": ["hello"]})
        mock_load = MagicMock(return_value=mock_ds)

        with patch("datasets.load_dataset", mock_load):
            load_dataset(
                "squad",
                storage=temp_storage,
                split="train[:100]",
                trust_remote_code=True,
            )

        mock_load.assert_called_once_with(
            "squad",
            split="train[:100]",
            trust_remote_code=True,
        )


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataset(self, temp_storage):
        """Test handling empty dataset."""
        empty_ds = datasets.Dataset.from_dict({"x": []})
        local_ds = LocalDataset.from_hf(empty_ds, "test", temp_storage)

        assert local_ds.row_count == 0
        table = local_ds.load()
        assert table.num_rows == 0

    def test_single_row(self, temp_storage):
        """Test handling single row dataset."""
        single_ds = datasets.Dataset.from_dict({"x": [42]})
        local_ds = LocalDataset.from_hf(single_ds, "test", temp_storage)

        assert local_ds.row_count == 1
        hf_ds = local_ds.to_hf()
        assert hf_ds["x"] == [42]

    def test_publish_not_implemented(self, sample_hf_dataset, temp_storage):
        """Test that publish() raises NotImplementedError."""
        local_ds = LocalDataset.from_hf(sample_hf_dataset, "test", temp_storage)

        with pytest.raises(NotImplementedError):
            local_ds.publish()

    def test_repr(self, sample_hf_dataset, temp_storage):
        """Test __repr__ method."""
        local_ds = LocalDataset.from_hf(sample_hf_dataset, "test", temp_storage)

        repr_str = repr(local_ds)
        assert "LocalDataset" in repr_str
        assert "test" in repr_str
        assert "0.1.0" in repr_str
