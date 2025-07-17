import typing as t
from pathlib import Path
from typing import ClassVar

from dreadnode.data_types.base import DataType

Object3DDataType = str | Path | bytes


class Object3D(DataType):
    """
    3D object media type for Dreadnode logging.

    Supports:
    - Local file paths to 3D models (.obj, .glb, .gltf, etc.)
    - Raw bytes with metadata
    """

    SUPPORTED_FORMATS: ClassVar[list[str]] = [
        "obj",
        "glb",
        "gltf",
        "stl",
        "fbx",
        "ply",
        "dae",
        "usdz",
    ]

    def __init__(
        self,
        data: Object3DDataType,
        caption: str | None = None,
        format: str | None = None,
    ):
        """
        Initialize a 3D Object.

        Args:
            data: The 3D object data, which can be:
                - A path to a local 3D model file (str or Path)
                - Raw bytes of a 3D model file
            caption: Optional caption for the 3D object
            format: Optional format override (obj, glb, etc.)
        """
        self._data = data
        self._caption = caption
        self._format = format

    def to_serializable(self) -> tuple[bytes, dict[str, t.Any]]:
        """
        Convert the 3D object to bytes and return with metadata.

        Returns:
            A tuple of (object_bytes, metadata_dict)
        """
        if isinstance(self._data, (str, Path)) and Path(self._data).exists():
            return self._process_file_path()
        if isinstance(self._data, bytes):
            format_name = self._format or "glb"
            return self._data, self._generate_metadata(format_name)
        raise TypeError(f"Unsupported 3D object data type: {type(self._data)}")

    def _process_file_path(self) -> tuple[bytes, dict[str, t.Any]]:
        """
        Process a 3D object from a file path.
        Returns:
            A tuple of (object_bytes, metadata_dict)
        """
        if not isinstance(self._data, (str, Path)):
            raise TypeError(f"Expected str or Path for file path, got {type(self._data)}")
        path = Path(self._data)
        object_bytes = path.read_bytes()
        format_name = self._format or path.suffix.lstrip(".")

        metadata = self._generate_metadata(format_name)
        return object_bytes, metadata

    def _generate_metadata(self, format_name: str) -> dict[str, t.Any]:
        """
        Generate metadata for the 3D object.
        Args:
            format_name: The format of the 3D object (obj, glb, etc.)
        Returns:
            A dictionary of metadata
        """
        metadata = {
            "extension": format_name.lower(),
            "x-python-datatype": "dreadnode.Object3D.bytes",
        }

        if self._caption:
            metadata["caption"] = self._caption

        if isinstance(self._data, (str, Path)):
            metadata["source-type"] = "file"
            metadata["source-path"] = str(self._data)
        elif isinstance(self._data, bytes):
            metadata["source-type"] = "bytes"

        return metadata
