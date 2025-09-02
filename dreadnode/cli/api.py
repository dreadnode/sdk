import atexit
import base64
import json
from datetime import datetime, timezone

from dreadnode.api.client import ApiClient
from dreadnode.constants import (
    DEFAULT_TOKEN_MAX_TTL,
)
from dreadnode.user_config import UserConfig


class Token:
    """A JWT token with an expiration time."""

    data: str
    expires_at: datetime

    @staticmethod
    def parse_jwt_token_expiration(token: str) -> datetime:
        """Return the expiration date from a JWT token."""

        _, b64payload, _ = token.split(".")
        payload = base64.urlsafe_b64decode(b64payload + "==").decode("utf-8")
        return datetime.fromtimestamp(json.loads(payload).get("exp"), tz=timezone.utc)

    def __init__(self, token: str):
        self.data = token
        self.expires_at = Token.parse_jwt_token_expiration(token)

    def ttl(self) -> int:
        """Get number of seconds left until the token expires."""
        return int((self.expires_at - datetime.now(tz=timezone.utc)).total_seconds())

    def is_expired(self) -> bool:
        """Return True if the token is expired."""
        return self.ttl() <= 0

    def is_close_to_expiry(self) -> bool:
        """Return True if the token is close to expiry."""
        return self.ttl() <= DEFAULT_TOKEN_MAX_TTL


def create_api_client(*, profile: str | None = None) -> ApiClient:
    """Create an authenticated API client using stored configuration data."""

    user_config = UserConfig.read()
    config = user_config.get_server_config(profile)

    client = ApiClient(
        config.url,
        cookies={
            "access_token": config.access_token,
            "refresh_token": config.refresh_token,
        },
    )

    # Preemptively check if the token is expired
    if Token(config.refresh_token).is_expired():
        raise RuntimeError("Authentication expired, use [bold]dreadnode login[/]")

    def _flush_auth_changes() -> None:
        """Flush the authentication data to disk if it has been updated."""

        access_token = client._client.cookies.get("access_token")  # noqa: SLF001
        refresh_token = client._client.cookies.get("refresh_token")  # noqa: SLF001

        changed: bool = False
        if access_token and access_token != config.access_token:
            changed = True
            config.access_token = access_token

        if refresh_token and refresh_token != config.refresh_token:
            changed = True
            config.refresh_token = refresh_token

        if changed:
            user_config.set_server_config(config, profile).write()

    atexit.register(_flush_auth_changes)

    return client
