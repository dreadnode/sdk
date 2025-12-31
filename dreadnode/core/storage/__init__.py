from fsspec import AbstractFileSystem, register_implementation

from dreadnode.core.packaging.loader import BaseLoader
from dreadnode.core.packaging.package import Package
from dreadnode.core.storage.storage import Storage


class DreadnodeFS(AbstractFileSystem):  # type: ignore[misc]
    protocol = ("dn", "dreadnode")


register_implementation("dreadnode", DreadnodeFS)
register_implementation("dn", DreadnodeFS)


__all__ = [
    "BaseLoader",
    "Package",
    "Storage",
]
