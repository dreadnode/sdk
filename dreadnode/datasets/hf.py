"""HuggingFace datasets integration utilities."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datasets
    import pyarrow as pa


def require_datasets() -> None:
    """Raise ModuleNotFoundError if datasets package is not installed."""
    if importlib.util.find_spec("datasets") is None:
        raise ModuleNotFoundError(
            "The 'datasets' package is required for HuggingFace integration. "
            "Install it with: pip install dreadnode[training]"
        )


def arrow_to_hf(table: pa.Table) -> datasets.Dataset:
    """Convert PyArrow Table to HuggingFace Dataset.

    This is a zero-copy operation when possible, as HuggingFace
    datasets are backed by Arrow tables internally.

    Args:
        table: PyArrow Table to convert.

    Returns:
        HuggingFace Dataset wrapping the table.
    """
    require_datasets()
    import datasets

    return datasets.Dataset(table)


def hf_to_arrow(hf_dataset: datasets.Dataset) -> pa.Table:
    """Get the underlying PyArrow Table from a HuggingFace Dataset.

    Args:
        hf_dataset: HuggingFace Dataset to convert.

    Returns:
        The underlying PyArrow Table.
    """
    return hf_dataset.data.table


def infer_schema(hf_dataset: datasets.Dataset) -> dict[str, str]:
    """Extract schema from HuggingFace Dataset in manifest format.

    Args:
        hf_dataset: HuggingFace Dataset to extract schema from.

    Returns:
        Dictionary mapping column names to type strings.
    """
    schema = {}
    for name, feature in hf_dataset.features.items():
        schema[name] = str(feature)
    return schema
