from pathlib import Path

from pydantic import BaseModel, field_serializer

from dreadnode.api.models import RegistryImageDetails
from dreadnode.cli.platform.constants import (
    API_SERVICE,
    UI_SERVICE,
    VERSION_CONFIG_PATH,
)
from dreadnode.cli.platform.tag import tag_to_semver


class LocalVersion(RegistryImageDetails):
    local_path: Path
    current: bool

    def __str__(self) -> str:
        return self.tag

    @field_serializer("local_path")
    def serialize_path(self, path: Path) -> str:
        """Serialize Path object to absolute path string.

        Args:
            path: Path object to serialize.

        Returns:
            str: Absolute path as string.
        """
        return str(path.resolve())  # Convert to absolute path string

    @property
    def details(self) -> str:
        configured_overrides = (
            "\n".join(
                f"  - {line}" for line in self.configure_overrides_env_file.read_text().splitlines()
            )
            if self.configure_overrides_env_file.exists()
            else "  (none)"
        )

        return (
            f"Tag: {self.tag}\n"
            f"Local Path: {self.local_path}\n"
            f"Compose File: {self.compose_file}\n"
            f"API Env File: {self.api_env_file}\n"
            f"UI Env File: {self.ui_env_file}\n"
            f"Configured: \n{configured_overrides}\n"
        )

    @property
    def compose_file(self) -> Path:
        return self.local_path / "docker-compose.yaml"

    @property
    def api_env_file(self) -> Path:
        return self.local_path / f".{API_SERVICE}.env"

    @property
    def api_example_env_file(self) -> Path:
        return self.local_path / f".{API_SERVICE}.example.env"

    @property
    def ui_env_file(self) -> Path:
        return self.local_path / f".{UI_SERVICE}.env"

    @property
    def ui_example_env_file(self) -> Path:
        return self.local_path / f".{UI_SERVICE}.example.env"

    @property
    def configure_overrides_env_file(self) -> Path:
        return self.local_path / ".configure.overrides.env"

    @property
    def configure_overrides_compose_file(self) -> Path:
        return self.local_path / "docker-compose.configure.overrides.yaml"

    @property
    def arg_overrides_env_file(self) -> Path:
        return self.local_path / ".arg.overrides.env"

    def get_env_path_by_service(self, service: str) -> Path:
        """Get environment file path for a specific service.

        Args:
            service: Service name to get env path for.

        Returns:
            Path: Path to the service's environment file.

        Raises:
            ValueError: If service is not recognized.
        """
        if service == API_SERVICE:
            return self.api_env_file
        if service == UI_SERVICE:
            return self.ui_env_file
        raise ValueError(f"Unknown service: {service}")

    def get_example_env_path_by_service(self, service: str) -> Path:
        """Get example environment file path for a specific service.

        Args:
            service: Service name to get example env path for.

        Returns:
            Path: Path to the service's example environment file.

        Raises:
            ValueError: If service is not recognized.
        """
        if service == API_SERVICE:
            return self.api_example_env_file
        if service == UI_SERVICE:
            return self.ui_example_env_file
        raise ValueError(f"Unknown service: {service}")


class VersionConfig(BaseModel):
    versions: list[LocalVersion]

    @classmethod
    def read(cls) -> "VersionConfig":
        """Read the version configuration from the file system or return an empty instance."""

        if not VERSION_CONFIG_PATH.exists():
            return cls(versions=[])

        with VERSION_CONFIG_PATH.open("r") as f:
            return cls.model_validate_json(f.read())

    def write(self) -> None:
        """Write the versions configuration to the file system."""

        if not VERSION_CONFIG_PATH.parent.exists():
            VERSION_CONFIG_PATH.parent.mkdir(parents=True)

        with VERSION_CONFIG_PATH.open("w") as f:
            f.write(self.model_dump_json())

    def add_version(self, version: LocalVersion) -> None:
        """
        Add a new version to the configuration if it doesn't already exist.

        Args:
            version: The LocalVersion instance to add.
        """
        if next((v for v in self.versions if v.tag == version.tag), None) is None:
            self.versions.append(version)
            self.write()

    def get_current_version(self, *, tag: str | None = None) -> LocalVersion | None:
        """Get the current active version or a specific version by tag."""
        if tag:
            return next((v for v in self.versions if v.tag == tag), None)

        if current := next((v for v in self.versions if v.current), None):
            return current

        if latest := self.get_latest_version():
            self.set_current_version(latest)
            return latest

        return None

    def get_latest_version(self) -> LocalVersion | None:
        """Get the latest version based on semantic versioning."""
        if not self.versions:
            return None
        sorted_versions = sorted(
            self.versions,
            key=lambda v: tag_to_semver(v.tag),
            reverse=True,
        )
        return sorted_versions[0]

    def get_by_tag(self, tag: str) -> LocalVersion:
        """
        Get a specific local platform version by its tag.

        Args:
            tag: The tag of the version to retrieve.

        Returns:
            LocalVersion: The version schema matching the provided tag.

        Raises:
            ValueError: If no version with the specified tag is found.
        """
        for version in self.versions:
            if version.tag == tag:
                return version
        raise ValueError(f"No local version found with tag: {tag}")

    def set_current_version(self, version: LocalVersion) -> None:
        """
        Mark a specific version as the current active version.

        Updates the versions manifest to mark the specified version as current
        and all others as not current.

        Args:
            version: The version to mark as current.
        """
        if next((v for v in self.versions if v.tag == version.tag), None) is None:
            self.versions.append(version)

        for available_version in self.versions:
            if available_version.tag == version.tag:
                available_version.current = True
            else:
                available_version.current = False

        self.write()
