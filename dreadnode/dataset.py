import uuid
from typing import Any

from fsspec import AbstractFileSystem
from pyarrow import Table, dataset
from pydantic import BaseModel, ConfigDict

from dreadnode.constants import DATASETS_CACHE


class DatasetMetadata(BaseModel):
    """
    A data model representing the metadata of a dataset.
    """

    name: str
    description: str | None = None
    version: str | None = None
    license: str | None = None
    tags: list[str] | None = None
    uri: str | None = None
    ds_schema: dict[str, Any] | None = None
    files: list[str] | None = None


class Dataset(BaseModel):
    """
    A data model representing a versioned dataset.

    """

    ds: dataset.Dataset | Table
    name: str | None = None
    description: str | None = None
    version: str | None = None
    license: str | None = None
    tags: list[str] | None = None

    _metadata: DatasetMetadata | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, _context: Any) -> None:
        """Post-initialization to handle private attributes."""

        if self._metadata:
            return

        if not self.name:
            self.name = uuid.uuid4().hex

        if not self.version:
            self.version = "0.0.1"

    @property
    def uri(self) -> str:
        """Get the URI of the dataset if available."""
        return self._get_uri()

    @property
    def ds_schema(self) -> dict:
        """Get the schema of the dataset."""
        return self._get_schema()

    @property
    def metadata(self) -> dict:
        """Get the manifest of the dataset."""
        return self._get_metadata()

    @property
    def files(self) -> list[str]:
        """Get the list of files in the dataset."""
        return self._get_files()

    def _get_uri(self) -> None:
        """Get a the uri for the dataset."""
        # normalize the name to create a uri
        return self.name.replace(" ", "_").lower()

    def _get_schema(self) -> dict[str, Any]:
        """Set the schema of the dataset."""
        if self.ds is None:
            raise ValueError("Dataset data is not loaded.")

        _ds_schema = {"fields": []}
        try:
            for name, dtype in zip(self.ds.schema.names, self.ds.schema.types, strict=False):
                _ds_schema["fields"].append({"name": name, "type": str(dtype)})
        except Exception as e:
            raise ValueError("Failed to extract schema from dataset.") from e

        return _ds_schema

    def _get_files(self) -> list[str]:
        """Set the list of files in the dataset."""
        if not self.ds:
            raise ValueError("Dataset data (ds) is not loaded.")
        return self.ds.files

    def _get_metadata(self) -> None:
        """Create a manifest for the dataset."""
        return DatasetMetadata(
            name=self.name,
            description=self.description,
            version=self.version,
            license=self.license,
            tags=self.tags,
            uri=self.uri,
            ds_schema=self.ds_schema,
            files=self.files,
        )

    def save(self, path: str, *, _fs: AbstractFileSystem) -> None:
        """
        Save the dataset to the given path.
        """

        dataset.write_dataset(data=self.ds, base_dir=path, format="parquet", filesystem=_fs)

    @classmethod
    def from_path(cls, path: str, *, _fs: AbstractFileSystem) -> "Dataset":
        """
        Create a Dataset instance from a given path.

        Args:
        path: The base path to the dataset.
        lazy: If True, loads a `pyarrow.dataset.Dataset` pointer without
              reading data into memory. If False, loads the data
              into an in-memory `pyarrow.Table`.
        Returns:
            A new Dataset instance.
        """

        ds_obj = dataset.dataset(path, format="parquet", filesystem=_fs)

        return cls(ds=ds_obj.to_table())

    @classmethod
    def from_json(cls, metadata: dict[str, Any]) -> "Dataset":
        """
        Create a Dataset instance from metadata.

        Args:
            metadata: The DatasetMetadata instance.
        Returns:
            A new Dataset instance.
        """

        try:
            valid_metadata = DatasetMetadata.model_validate(metadata, strict=True)
        except Exception as e:
            raise ValueError("Invalid dataset metadata.") from e

        base_data_path = f"{DATASETS_CACHE}/{valid_metadata.uri}/data"

        return cls(
            ds=dataset.dataset(base_data_path, format="parquet"),
            name=valid_metadata.name,
            description=valid_metadata.description,
            version=valid_metadata.version,
            license=valid_metadata.license,
            tags=valid_metadata.tags,
            metadata=valid_metadata,
        )
