import typing as t
from abc import ABC, abstractmethod


class DataType(ABC):
    """Base class for dedicated data types that can be logged with Dreadnode."""

    @abstractmethod
    def to_serializable(self) -> tuple[t.Any, dict[str, t.Any]]:
        """
        Convert the media type to a serializable format.

        Returns:
            Tuple of (data, metadata) where:
                - data: The serialized data
                - metadata: Additional metadata for this data type
        """


class WithMeta(DataType):
    """
    Helper data type to add additional metadata to the schema for logged data.

    Example:
        ```
        log_output("my_data", WithMeta(data, {"format": "custom-data"}))
        ```
    """

    def __init__(self, obj: t.Any, metadata: dict[str, t.Any]):
        """
        Initialize a data type with associated metadata.

        Args:
            metadata: The metadata for this data type
        """
        self._obj = obj
        self._metadata = metadata

    def to_serializable(self) -> tuple[t.Any, dict[str, t.Any]]:
        """
        Convert the media type to a serializable format.

        Returns:
            Tuple of (data, metadata) where:
                - data: The serialized data
                - metadata: Additional metadata for this data type
        """
        return self._obj, self._metadata
