import fsspec

from dreadnode.dataset import Dataset
from dreadnode.storage.datasets.metadata import DatasetMetadata
from dreadnode.storage.dnfs import DreadnodeFS

__all__ = [
    "Dataset",
    "DatasetMetadata",
    "load_dataset",
]

fsspec.register_implementation("dn", DreadnodeFS)
fsspec.register_implementation("dreadnode", DreadnodeFS)
