import abc
from pathlib import Path

from dreadnode.constants import DEFAULT_LOCAL_OBJECT_DIR


class ObjectStore(abc.ABC):
    @abc.abstractmethod
    async def store(self, data: bytes, hash: str) -> str: ...

    @abc.abstractmethod
    def exists(self, uri: str) -> bool: ...

    @abc.abstractmethod
    async def get(self, uri: str) -> bytes: ...


class LocalObjectStore(ObjectStore):
    def __init__(self, base_path: Path | str = DEFAULT_LOCAL_OBJECT_DIR):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve_uri(self, uri: str) -> Path:
        if not uri.startswith("file://"):
            raise ValueError(f"Invalid uri for local storage: {uri}")
        uri_without_scheme = uri[7:]
        return Path(uri_without_scheme)

    async def store(self, data: bytes, hash: str) -> str:
        path = self.base_path / hash
        path.write_bytes(data)
        return path.absolute().as_uri()

    def exists(self, uri: str) -> bool:
        if not uri.startswith("file://"):
            return False
        return self._resolve_uri(uri).exists()

    async def get(self, uri: str) -> bytes:
        return self._resolve_uri(uri).read_bytes()
