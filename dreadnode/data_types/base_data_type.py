import typing as t
from abc import ABC, abstractmethod


class BaseDataType(ABC):
    """Base class for all data types that can be logged with Dreadnode."""

    @abstractmethod
    def to_serializable(self) -> tuple[t.Any, dict[str, t.Any]]:
        """
        Convert the media type to a serializable format.

        Returns:
            Tuple of (data, metadata) where:
                - data: The serialized data
                - metadata: Additional metadata for this data type
        """
