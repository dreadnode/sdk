import fsspec

from dreadnode.storage.dnfs import DreadnodeS3FS

fsspec.register_implementation("dn", DreadnodeS3FS)
fsspec.register_implementation("dreadnode", DreadnodeS3FS)
