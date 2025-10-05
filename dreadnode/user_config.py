from pydantic import BaseModel
from ruamel.yaml import YAML

from dreadnode.constants import DEFAULT_PROFILE_NAME, USER_CONFIG_PATH
from dreadnode.logging_ import print_info


class ServerConfig(BaseModel):
    """Server specific authentication data and API URL."""

    url: str
    email: str
    username: str
    api_key: str
    access_token: str
    refresh_token: str


class UserConfig(BaseModel):
    """User configuration supporting multiple server profiles."""

    active: str | None = None
    servers: dict[str, ServerConfig] = {}

    def _update_active(self) -> None:
        """If active is not set, set it to the first available server and raise an error if no servers are configured."""

        if self.active not in self.servers:
            self.active = next(iter(self.servers)) if self.servers else None

    def _update_urls(self) -> bool:
        updated = False
        for search, replace in {
            "//staging-crucible.dreadnode.io": "//staging-platform.dreadnode.io",
            "//dev-crucible.dreadnode.io": "//dev-platform.dreadnode.io",
            "//crucible.dreadnode.io": "//platform.dreadnode.io",
        }.items():
            for server in self.servers.values():
                if search in server.url:
                    server.url = server.url.replace(search, replace)
                    updated = True
        return updated

    @classmethod
    def read(cls) -> "UserConfig":
        """Read the user configuration from the file system or return an empty instance."""

        if not USER_CONFIG_PATH.exists():
            return cls()

        with USER_CONFIG_PATH.open("r") as f:
            self = cls.model_validate(YAML().load(f))

        if self._update_urls():
            self.write()

        return self

    def write(self) -> None:
        """Write the user configuration to the file system."""

        self._update_active()

        if not USER_CONFIG_PATH.parent.exists():
            print_info(f"Creating config at {USER_CONFIG_PATH.parent}")
            USER_CONFIG_PATH.parent.mkdir(parents=True)

        with USER_CONFIG_PATH.open("w") as f:
            YAML().dump(self.model_dump(mode="json"), f)

    @property
    def active_profile_name(self) -> str | None:
        """Get the name of the active profile."""
        self._update_active()
        return self.active

    def get_server_config(self, profile: str | None = None) -> ServerConfig:
        """Get the server configuration for the given profile or None if not set."""

        profile = profile or self.active
        if not profile:
            raise RuntimeError("No profile is set, use [bold]dreadnode login[/] to authenticate")

        if profile not in self.servers:
            raise RuntimeError(f"No server configuration for profile: {profile}")

        return self.servers[profile]

    def set_server_config(self, config: ServerConfig, profile: str | None = None) -> "UserConfig":
        """Set the server configuration for the given profile."""

        profile = profile or self.active or DEFAULT_PROFILE_NAME
        self.servers[profile] = config
        return self


def is_dreadnode_saas_server(url: str) -> bool:
    """Check if the server URL is a Dreadnode SaaS server (ends with dreadnode.io)."""
    return url.rstrip("/").endswith(".dreadnode.io")


def find_dreadnode_saas_profiles(user_config: UserConfig) -> list[str]:
    """Find all profiles that point to Dreadnode SaaS servers."""
    saas_profiles = []
    for profile_name, server_config in user_config.servers.items():
        if is_dreadnode_saas_server(server_config.url):
            saas_profiles.append(profile_name)
    return saas_profiles
