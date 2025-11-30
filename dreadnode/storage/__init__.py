from fsspec import AbstractFileSystem, register_implementation

from dreadnode.dataset import Dataset, load_dataset


class DreadnodeFS(AbstractFileSystem):
    protocol = ("dn", "dreadnode")


register_implementation("dreadnode", DreadnodeFS)
register_implementation("dn", DreadnodeFS)

__all__ = [
    "Dataset",
    "load_dataset",
]
