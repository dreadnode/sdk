import typing as t
from pathlib import Path

import yaml
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from dreadnode.version import VERSION

# Type aliases
PlatformService = t.Literal["dreadnode-api", "dreadnode-ui"]
SupportedArchitecture = t.Literal["amd64", "arm64"]

PLATFORM_SERVICES = t.cast("list[PlatformService]", t.get_args(PlatformService))
API_SERVICE: PlatformService = "dreadnode-api"
UI_SERVICE: PlatformService = "dreadnode-ui"
SUPPORTED_ARCHITECTURES = t.cast("list[SupportedArchitecture]", t.get_args(SupportedArchitecture))

# Supported dataset file formats (constant, not configurable)
SUPPORTED_FORMATS = {
    "parquet": "parquet",
    "json": "json",
    "jsonl": "json",
    "csv": "csv",
    "arrow": "arrow",
    "ipc": "arrow",
}

# File name constants
METADATA_FILE = "metadata.json"
MANIFEST_FILE = "manifest.json"


DEFAULT_CACHE_DIR = Path.home() / ".dreadnode"
DEFAULT_REMOTE_STORAGE_PREFIX = "user-data"
DEFAULT_PROFILE_NAME = "main"
DEFAULT_POLL_INTERVAL = 5
DEFAULT_MAX_POLL_TIME = 300
DEFAULT_TOKEN_MAX_TTL = 60
DEFAULT_MAX_INLINE_OBJECT_BYTES = 10 * 1024  # 10KB
DEFAULT_PLATFORM_BASE_DOMAIN = "dreadnode.io"
DEFAULT_DOCKER_REGISTRY_SUBDOMAIN = "registry"
DEFAULT_DOCKER_REGISTRY_LOCAL_PORT = 5005
DEFAULT_DOCKER_REGISTRY_IMAGE_TAG = "registry"
DEFAULT_WORKSPACE_NAME = "Personal Workspace"
DEFAULT_PROJECT_NAME = "Default"
DEFAULT_PROJECT_KEY = "default"
DEFAULT_DOCKER_PROJECT_NAME = "dreadnode-platform"

# Storage constants
DEFAULT_CHUNK_SIZE = 64 * 1024 * 1024  # 64MB
DEFAULT_MAX_WORKERS = 10
DEFAULT_MULTIPART_THRESHOLD = 100 * 1024 * 1024  # 100MB
DEFAULT_MULTIPART_CHUNKSIZE = 64 * 1024 * 1024  # 64MB
DEFAULT_FS_CREDENTIAL_REFRESH_BUFFER = 900  # 15 minutes

# YAML config file name
SETTINGS_FILE = "settings.yaml"


def get_cache_object_dir(cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return cache_dir / "objects"


def get_user_config_path(cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return cache_dir / "config"


def get_platform_storage_dir(cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return cache_dir / "platform"


def get_version_config_path(cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return get_platform_storage_dir(cache_dir) / "versions.json"


def get_settings_file_path(cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    return cache_dir / SETTINGS_FILE


def load_yaml_settings(settings_path: Path | None = None) -> dict[str, t.Any]:
    """
    Load settings from YAML configuration file.

    Args:
        settings_path: Optional path to settings file. If not provided,
                      uses the default location (~/.dreadnode/settings.yaml).

    Returns:
        Dictionary of settings from the YAML file, or empty dict if file doesn't exist.
    """
    if settings_path is None:
        settings_path = get_settings_file_path()

    if not settings_path.exists():
        return {}

    with settings_path.open() as f:
        data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source that loads from YAML file."""

    def __init__(
        self,
        settings_cls: type[BaseSettings],
        yaml_path: Path | None = None,
    ):
        super().__init__(settings_cls)
        self._yaml_data = load_yaml_settings(yaml_path)

    def get_field_value(
        self,
        field: t.Any,
        field_name: str,
    ) -> tuple[t.Any, str, bool]:
        """Get value for a field from YAML data."""
        value = self._yaml_data.get(field_name)
        return value, field_name, value is not None

    def __call__(self) -> dict[str, t.Any]:
        """Return all YAML settings."""
        return self._yaml_data


# Global variable to store YAML path for settings source
_yaml_settings_path: Path | None = None


class DreadnodeSettings(BaseSettings):
    """
    Settings loaded from environment variables and YAML configuration.

    Priority order (highest to lowest):
    1. Explicit parameters passed to functions
    2. Environment variables (DREADNODE_* prefix)
    3. YAML configuration file (~/.dreadnode/settings.yaml)
    4. Default values
    """

    model_config = SettingsConfigDict(
        env_prefix="DREADNODE_",
        extra="ignore",
    )

    # Server/Auth - with fallback aliases
    server: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DREADNODE_SERVER_URL", "DREADNODE_SERVER"),
    )
    token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DREADNODE_API_TOKEN", "DREADNODE_API_KEY"),
    )
    platform_base_domain: str = Field(default=DEFAULT_PLATFORM_BASE_DOMAIN)

    # Organization/Workspace/Project
    organization: str | None = Field(default=None)
    workspace: str | None = Field(default=None)
    project: str | None = Field(default=None)
    profile: str | None = Field(default=None)

    # Default names
    default_workspace_name: str = Field(default=DEFAULT_WORKSPACE_NAME)
    default_project_name: str = Field(default=DEFAULT_PROJECT_NAME)
    default_project_key: str = Field(default=DEFAULT_PROJECT_KEY)

    # Paths
    cache_dir: Path = Field(default=DEFAULT_CACHE_DIR)
    user_config_file: Path | None = Field(default=None)

    # Storage settings
    remote_storage_prefix: str = Field(default=DEFAULT_REMOTE_STORAGE_PREFIX)
    max_inline_object_bytes: int = Field(default=DEFAULT_MAX_INLINE_OBJECT_BYTES)
    chunk_size: int = Field(default=DEFAULT_CHUNK_SIZE)
    max_workers: int = Field(default=DEFAULT_MAX_WORKERS)
    multipart_threshold: int = Field(default=DEFAULT_MULTIPART_THRESHOLD)
    multipart_chunksize: int = Field(default=DEFAULT_MULTIPART_CHUNKSIZE)
    fs_credential_refresh_buffer: int = Field(default=DEFAULT_FS_CREDENTIAL_REFRESH_BUFFER)

    # Auth flow settings
    poll_interval: int = Field(default=DEFAULT_POLL_INTERVAL)
    max_poll_time: int = Field(default=DEFAULT_MAX_POLL_TIME)
    token_max_ttl: int = Field(default=DEFAULT_TOKEN_MAX_TTL)

    # Docker registry settings
    docker_registry_subdomain: str = Field(default=DEFAULT_DOCKER_REGISTRY_SUBDOMAIN)
    docker_registry_local_port: int = Field(default=DEFAULT_DOCKER_REGISTRY_LOCAL_PORT)
    docker_registry_image_tag: str = Field(default=DEFAULT_DOCKER_REGISTRY_IMAGE_TAG)
    docker_project_name: str = Field(default=DEFAULT_DOCKER_PROJECT_NAME)

    # Flags
    debug: bool = Field(default=False)
    console: bool = Field(default=False)

    # Computed properties
    @property
    def server_url(self) -> str:
        """Get server URL with fallback to default."""
        return self.server or f"https://platform.{self.platform_base_domain}"

    @property
    def user_config_path(self) -> Path:
        """Get user config path with fallbacks."""
        if self.user_config_file:
            return self.user_config_file
        return get_user_config_path(self.cache_dir)

    @property
    def cache_object_dir(self) -> Path:
        """Local directory for dreadnode objects."""
        return self.cache_dir / "objects"

    @property
    def platform_storage_dir(self) -> Path:
        """Platform storage directory."""
        return self.cache_dir / "platform"

    @property
    def version_config_path(self) -> Path:
        """Path to versions config file."""
        return self.platform_storage_dir / "versions.json"

    @property
    def settings_file_path(self) -> Path:
        """Path to YAML settings file."""
        return self.cache_dir / SETTINGS_FILE

    @property
    def user_agent(self) -> str:
        """Default User-Agent string."""
        return f"dreadnode/{VERSION}"

    @property
    def has_env_credentials(self) -> bool:
        """Check if credentials are set via environment."""
        return bool(self.server or self.token)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        Customize settings sources priority.

        Priority order (highest to lowest):
        1. Init settings (constructor arguments)
        2. Environment variables
        3. YAML configuration file
        4. Default values
        """
        global _yaml_settings_path
        return (
            init_settings,
            env_settings,
            YamlSettingsSource(settings_cls, _yaml_settings_path),
        )


def create_settings(yaml_path: Path | None = None) -> DreadnodeSettings:
    """
    Create settings instance with YAML config as defaults.

    The priority order is:
    1. Environment variables (handled by pydantic-settings)
    2. YAML configuration file
    3. Default values

    Args:
        yaml_path: Optional path to YAML settings file.

    Returns:
        Configured DreadnodeSettings instance.
    """
    global _yaml_settings_path
    _yaml_settings_path = yaml_path
    return DreadnodeSettings()


# Singleton instance for settings
settings = create_settings()


# Backward compatibility aliases
DreadnodeEnvSettings = DreadnodeSettings

DEBUG = settings.debug
PLATFORM_BASE_URL = settings.server_url
USER_CONFIG_PATH = settings.user_config_path
DEFAULT_CACHE_OBJECT_DIR = str(settings.cache_object_dir)
PLATFORM_STORAGE_DIR = settings.platform_storage_dir
VERSION_CONFIG_PATH = settings.version_config_path
DEFAULT_USER_AGENT = settings.user_agent

# Storage constants (now configurable via env/yaml)
CHUNK_SIZE = settings.chunk_size
MAX_WORKERS = settings.max_workers
MULTIPART_THRESHOLD = settings.multipart_threshold
MULTIPART_CHUNKSIZE = settings.multipart_chunksize
FS_CREDENTIAL_REFRESH_BUFFER = settings.fs_credential_refresh_buffer
