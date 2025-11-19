import fsspec

from dreadnode.dataset import Dataset
from dreadnode.storage.datasets.metadata import DatasetMetadata, DatasetSchema
from dreadnode.storage.dnfs import DreadnodeS3FS

__version__ = "0.1.0"

__all__ = [
    "Dataset",
    "DatasetMetadata",
    "DatasetSchema",
    "load_dataset",
]

fsspec.register_implementation("dn", DreadnodeS3FS)
fsspec.register_implementation("dreadnode", DreadnodeS3FS)
