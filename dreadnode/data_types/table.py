import io
import typing as t
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from dreadnode.data_types.base import DataType

TableDataType = (
    pd.DataFrame | dict[t.Any, t.Any] | list[t.Any] | str | Path | np.ndarray[t.Any, t.Any]
)


class Table(DataType):
    """
    Table data type for Dreadnode logging.

    Supports:
    - Pandas DataFrames
    - CSV/Parquet/JSON files
    - Dict or list data structures
    - NumPy arrays
    """

    SUPPORTED_FORMATS: ClassVar[list[str]] = ["csv", "parquet", "json"]

    def __init__(
        self,
        data: TableDataType,
        caption: str | None = None,
        format: str | None = None,
        *,
        index: bool = False,
    ):
        """
        Initialize a Table object.

        Args:
            data: The table data, which can be:
                - A pandas DataFrame
                - A path to a CSV/JSON/Parquet file
                - A dict or list of dicts
                - A NumPy array
            caption: Optional caption for the table
            format: Optional format to use when saving (csv, parquet, json)
            index: Whether to include index in the output
        """
        self._data = data
        self._caption = caption
        self._format = format or "csv"  # Default to CSV
        if self._format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format: {self._format}. "
                f"Supported formats are: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        self._index = index

    def to_serializable(self) -> tuple[bytes, dict[str, t.Any]]:
        """
        Convert the table to bytes and return with metadata.

        Returns:
            A tuple of (table_bytes, metadata_dict)
        """
        data_frame = self._to_dataframe()

        table_bytes = self._dataframe_to_bytes(data_frame)
        metadata = self._generate_metadata(data_frame)

        return table_bytes, metadata

    def _to_dataframe(self) -> pd.DataFrame:
        """
        Convert the input data to a pandas DataFrame.
        Returns:
            A pandas DataFrame representation of the input data
        """
        if isinstance(self._data, pd.DataFrame):
            return self._data
        if isinstance(self._data, (str, Path)) and Path(self._data).exists():
            path = Path(self._data)
            suffix = path.suffix.lower()

            if suffix == ".csv":
                return pd.read_csv(path)
            if suffix == ".parquet":
                return pd.read_parquet(path)
            if suffix in (".json", ".jsonl"):
                return pd.read_json(path)
            raise ValueError(f"Unsupported file format: {suffix}")

        if isinstance(self._data, dict):
            return pd.DataFrame.from_dict(self._data)

        if isinstance(self._data, (list, np.ndarray)):
            return pd.DataFrame(self._data)

        raise ValueError(f"Unsupported table data type: {type(self._data)}")

    def _dataframe_to_bytes(self, data_frame: pd.DataFrame) -> bytes:
        """
        Convert the DataFrame to bytes based on the specified format.
        Args:
            data_frame: The pandas DataFrame to convert
        Returns:
            Bytes representation of the DataFrame
        """
        buffer = io.BytesIO()

        if self._format == "csv":
            data_frame.to_csv(buffer, index=self._index)
        elif self._format == "parquet":
            data_frame.to_parquet(buffer, index=self._index)
        elif self._format == "json":
            json_str = data_frame.to_json(orient="records")
            buffer.write(json_str.encode())
        else:
            data_frame.to_csv(buffer, index=self._index)

        buffer.seek(0)
        return buffer.getvalue()

    def _generate_metadata(self, data_frame: pd.DataFrame) -> dict[str, t.Any]:
        """
        Generate metadata for the table.
        Args:
            data_frame: The pandas DataFrame to generate metadata for
        Returns:
            A dictionary of metadata
        """
        metadata = {
            "extension": self._format,
            "x-python-datatype": "dreadnode.Table.bytes",
            "rows": len(data_frame),
            "columns": len(data_frame.columns),
        }

        metadata["column-names"] = data_frame.columns.tolist()

        if self._caption:
            metadata["caption"] = self._caption

        if isinstance(self._data, pd.DataFrame):
            metadata["source-type"] = "pandas.DataFrame"
        elif isinstance(self._data, (str, Path)):
            metadata["source-type"] = "file"
            metadata["source-path"] = str(self._data)
        elif isinstance(self._data, dict):
            metadata["source-type"] = "dict"
        elif isinstance(self._data, list):
            metadata["source-type"] = "list"
        elif isinstance(self._data, np.ndarray):
            metadata["source-type"] = "numpy.ndarray"

        return metadata
