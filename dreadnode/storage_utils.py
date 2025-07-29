import functools
import typing as t

from dreadnode.util import logger


def with_credential_refresh(func: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
    """Decorator that automatically handles credential refresh on storage errors."""

    @functools.wraps(func)
    def wrapper(self: t.Any, *args: t.Any, **kwargs: t.Any) -> t.Any:
        # Try to refresh credentials before operation
        if hasattr(self, "_refresh_credentials_if_needed"):
            self._refresh_credentials_if_needed()

        try:
            return func(self, *args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if any(
                error in error_str
                for error in [
                    "ExpiredToken",
                    "TokenRefreshRequired",
                    "InvalidAccessKeyId",
                    "The Access Key Id you provided does not exist",
                ]
            ):
                logger.info("Storage credential error, forcing refresh and retrying")

                if hasattr(self, "_refresh_credentials_if_needed"):
                    self._refresh_credentials_if_needed()

                return func(self, *args, **kwargs)
            raise

    return wrapper
