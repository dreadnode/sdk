import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, TypeVar

from botocore.exceptions import ClientError  # type: ignore[import-untyped]
from loguru import logger

from dreadnode.constants import FS_CREDENTIAL_REFRESH_BUFFER
from dreadnode.util import resolve_endpoint

if TYPE_CHECKING:
    from s3fs import S3FileSystem  # type: ignore[import-untyped]

    from dreadnode.api.models import UserDataCredentials

T = TypeVar("T")


class CredentialManager:
    """Simple credential manager that handles S3 credential refresh automatically."""

    def __init__(self, credential_fetcher: Callable[[], "UserDataCredentials"]):
        """
        Initialize credential manager.

        Args:
            credential_fetcher: Function that returns new UserDataCredentials when called
        """
        self._credential_fetcher = credential_fetcher
        self._credentials: UserDataCredentials | None = None
        self._credentials_expiry: datetime | None = None
        self._filesystem: S3FileSystem | None = None
        self._prefix = ""

    def initialize(self) -> None:
        """Initialize with fresh credentials."""
        self._refresh_credentials()

    def get_filesystem(self) -> "S3FileSystem":
        """Get current filesystem, refreshing credentials if needed."""
        if self._needs_refresh():
            self._refresh_credentials()
        assert self._filesystem is not None  # noqa: S101
        return self._filesystem

    def get_prefix(self) -> str:
        """Get current prefix path."""
        return self._prefix

    def _needs_refresh(self) -> bool:
        """Check if credentials need refreshing."""
        if not self._credentials_expiry or not self._filesystem:
            return True

        now = datetime.now(timezone.utc)
        time_left = (self._credentials_expiry - now).total_seconds()
        return time_left < FS_CREDENTIAL_REFRESH_BUFFER

    def _refresh_credentials(self) -> None:
        """Refresh credentials and create new filesystem."""
        from s3fs import S3FileSystem

        try:
            logger.info("Refreshing storage credentials")
            new_credentials = self._credential_fetcher()
            resolved_endpoint = resolve_endpoint(new_credentials.endpoint)

            new_filesystem = S3FileSystem(
                key=new_credentials.access_key_id,
                secret=new_credentials.secret_access_key,
                token=new_credentials.session_token,
                client_kwargs={
                    "endpoint_url": resolved_endpoint,
                    "region_name": new_credentials.region,
                },
                use_listings_cache=False,
                listings_expiry_time=0,
                skip_instance_cache=True,
            )

            # Update internal state
            self._credentials = new_credentials
            self._credentials_expiry = new_credentials.expiration
            self._filesystem = new_filesystem
            self._prefix = f"{new_credentials.bucket}/{new_credentials.prefix}/"

            logger.info(f"Storage credentials refreshed, valid until {self._credentials_expiry}")

        except Exception:
            logger.exception("Failed to refresh storage credentials")
            raise

    def execute_with_retry(self, operation: Callable[[], T], max_retries: int = 3) -> T:
        """
        Execute an operation with automatic credential refresh on auth errors.

        Args:
            operation: Function to execute (should use self.get_filesystem())
            max_retries: Maximum number of retry attempts

        Returns:
            Result of the operation
        """
        for attempt in range(max_retries):
            try:
                return operation()
            except ClientError as e:  # noqa: PERF203
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code in ["ExpiredToken", "InvalidAccessKeyId", "SignatureDoesNotMatch"]:
                    logger.info(
                        "Credential error on attempt %d/%d, refreshing...", attempt + 1, max_retries
                    )

                    try:
                        self._refresh_credentials()
                    except Exception:
                        logger.exception("Failed to refresh credentials")
                        if attempt == max_retries - 1:
                            raise

                    if attempt < max_retries - 1:
                        time.sleep(attempt + 1)
                        continue
                else:
                    raise
            except Exception:
                raise

        raise RuntimeError(
            f"Operation failed after {max_retries} attempts due to credential issues"
        )
