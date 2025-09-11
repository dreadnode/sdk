from pathlib import Path

from pydantic import BaseModel, field_serializer

from dreadnode.api.models import RegistryImageDetails
from dreadnode.cli.platform.constants import API_SERVICE, UI_SERVICE


class LocalVersionSchema(RegistryImageDetails):
    local_path: Path
    current: bool

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
    def overrides_env_file(self) -> Path:
        return self.local_path / ".overrides.env"

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


class LocalVersionsSchema(BaseModel):
    versions: list[LocalVersionSchema]
