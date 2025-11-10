from collections.abc import Callable
from typing import TYPE_CHECKING

from dreadnode.storage.base import BaseStorage

if TYPE_CHECKING:
    from dreadnode.api.models import UserDataCredentials


class ArtifactStorage(BaseStorage):
    """
    Storage for artifacts with efficient handling of large files and directories.

    Supports:
    - Content-based deduplication using SHA1 hashing
    - Batch uploads for directories handled by fsspec
    """

    def __init__(self, credential_fetcher: Callable[[], "UserDataCredentials"]):
        """
        Initialize artifact storage with credential manager.

        Args:
            storage_manager: Optional credential manager for S3 operations
        """
        super().__init__(credential_fetcher=credential_fetcher)
