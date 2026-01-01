from __future__ import annotations

import contextlib
import os
import re
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel
from ulid import ULID

from dreadnode.core.api.models import Organization, Project, Workspace, WorkspaceFilter

if TYPE_CHECKING:
    from dreadnode.core.api.client import ApiClient

from dreadnode.core.log import logger
from dreadnode.core.util import create_key_from_name, valid_key
from dreadnode.core.settings import (
    DEFAULT_PROFILE_NAME,
    DEFAULT_PROJECT_KEY,
    DEFAULT_PROJECT_NAME,
    USER_CONFIG_PATH,
)


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
        """If active is not set, set it to the first available server."""
        if self.active not in self.servers:
            self.active = next(iter(self.servers)) if self.servers else None

    @property
    def active_profile_name(self) -> str | None:
        """Get the name of the active profile."""
        self._update_active()
        return self.active

    @classmethod
    def read(cls) -> UserConfig:
        """Read the user configuration from the file system or return an empty instance."""
        if not USER_CONFIG_PATH.exists():
            return cls()

        with USER_CONFIG_PATH.open("r") as f:
            return cls.model_validate(yaml.safe_load(f))

    def write(self) -> None:
        """Write the user configuration to the file system."""
        self._update_active()

        if not USER_CONFIG_PATH.parent.exists():
            USER_CONFIG_PATH.parent.mkdir(parents=True)

        with USER_CONFIG_PATH.open("w") as f:
            yaml.dump(self.model_dump(mode="json"), f)

    def get_server_config(self, profile: str | None = None) -> ServerConfig:
        """Get the server configuration for the given profile."""
        profile = profile or self.active
        if not profile:
            raise RuntimeError("No profile is set, use [bold]dreadnode login[/] to authenticate")

        if profile not in self.servers:
            raise RuntimeError(f"No server configuration for profile: {profile}")

        return self.servers[profile]

    def set_server_config(self, config: ServerConfig, profile: str | None = None) -> UserConfig:
        """Set the server configuration for the given profile."""
        profile = profile or self.active or DEFAULT_PROFILE_NAME
        self.servers[profile] = config
        return self

    def get_profile_server(self, profile: str | None = None) -> str | None:
        """Get the server URL from the user config for a given profile."""
        with contextlib.suppress(Exception):
            profile = profile or os.environ.get(DEFAULT_PROFILE_NAME)
            server_config = self.get_server_config(profile)
            return server_config.url
        return None

    def get_profile_api_key(self, profile: str | None = None) -> str | None:
        """Get the API key from the user config for a given profile."""
        with contextlib.suppress(Exception):
            profile = profile or os.environ.get(DEFAULT_PROFILE_NAME)
            server_config = self.get_server_config(profile)
            return server_config.api_key
        return None


class Session:
    """
    Session manages RBAC resolution for organization, workspace, and project.

    It encapsulates the logic for resolving and validating the user's access
    to organizations, workspaces, and projects.
    """

    def __init__(
        self,
        api: ApiClient,
        *,
        organization: str | ULID | None = None,
        workspace: str | ULID | None = None,
        project: str | None = None,
    ) -> None:
        self.api = api

        # Input identifiers (can be name, key, or ULID)
        self._organization_id: str | ULID | None = organization
        self._workspace_id: str | ULID | None = workspace
        self._project_id: str | None = project

        # Resolved objects
        self._organization: Organization | None = None
        self._workspace: Workspace | None = None
        self._project: Project | None = None

    @property
    def organization(self) -> Organization:
        """Get the resolved organization."""
        if self._organization is None:
            raise RuntimeError("Organization not resolved. Call resolve() first.")
        return self._organization

    @property
    def workspace(self) -> Workspace:
        """Get the resolved workspace."""
        if self._workspace is None:
            raise RuntimeError("Workspace not resolved. Call resolve() first.")
        return self._workspace

    @property
    def project(self) -> Project:
        """Get the resolved project."""
        if self._project is None:
            raise RuntimeError("Project not resolved. Call resolve() first.")
        return self._project

    @property
    def project_id(self) -> str:
        """Get the project ID as a string (for use in traces/spans)."""
        return str(self.project.id)

    def resolve(self) -> Session:
        """
        Resolve organization, workspace, and project.

        Returns:
            self for chaining.

        Raises:
            RuntimeError: If resolution fails.
        """
        self._resolve_organization()
        self._resolve_workspace()
        self._resolve_project()
        return self

    def _resolve_organization(self) -> None:
        """Resolve the organization to use based on configuration."""
        # Try to parse as ULID
        with contextlib.suppress(ValueError):
            self._organization_id = ULID(str(self._organization_id))

        if isinstance(self._organization_id, str) and not valid_key(self._organization_id):
            raise RuntimeError(
                f'Invalid Organization Key: "{self._organization_id}". '
                "The expected characters are lowercase letters, numbers, and hyphens (-)."
            )

        if self._organization_id:
            self._organization = self.api.get_organization(self._organization_id)
            if not self._organization:
                raise RuntimeError(f"Organization '{self._organization_id}' not found.")
        else:
            organizations = self.api.list_organizations()

            if not organizations:
                raise RuntimeError(
                    "You are not part of any organizations. You will not be able to use Strikes."
                )

            if len(organizations) > 1:
                org_list = "\n".join([f"- {o.key}" for o in organizations])
                raise RuntimeError(
                    f"You are part of multiple organizations. Please specify one:\n{org_list}"
                )
            self._organization = organizations[0]

    def _resolve_workspace(self) -> None:
        """Resolve the workspace to use based on configuration."""
        # Try to parse as ULID
        with contextlib.suppress(ValueError):
            self._workspace_id = ULID(str(self._workspace_id))

        if isinstance(self._workspace_id, str) and not valid_key(self._workspace_id):
            raise RuntimeError(
                f'Invalid Workspace Key: "{self._workspace_id}". '
                "The expected characters are lowercase letters, numbers, and hyphens (-)."
            )

        found_workspace: Workspace | None = None

        if self._workspace_id:
            try:
                found_workspace = self.api.get_workspace(
                    self._workspace_id, org_id=self.organization.id
                )
            except RuntimeError as e:
                if "404: Workspace not found" not in str(e):
                    raise

            if not found_workspace and isinstance(self._workspace_id, ULID):
                raise RuntimeError(f"Workspace with ID '{self._workspace_id}' not found.")

            if not found_workspace and isinstance(self._workspace_id, str):
                found_workspace = self._create_workspace(key=self._workspace_id)
        else:
            workspaces = self.api.list_workspaces(
                filters=WorkspaceFilter(org_id=self.organization.id)
            )
            found_workspace = next((ws for ws in workspaces if ws.is_default), None)

            if not found_workspace:
                raise RuntimeError("No default workspace found. Please specify a workspace.")

        if not found_workspace:
            raise RuntimeError("Failed to resolve or create a workspace.")

        self._workspace = found_workspace

    def _create_workspace(self, key: str) -> Workspace:
        """Create a new workspace."""
        try:
            logger.warning(f"Creating new workspace '{key}'")
            key = create_key_from_name(key)
            return self.api.create_workspace(
                name=key, key=key, organization_id=self.organization.id
            )
        except RuntimeError as e:
            if "403: Forbidden" in str(e):
                raise RuntimeError(
                    "You do not have permission to create workspaces for this organization."
                ) from e
            raise

    def _resolve_project(self) -> None:
        """Resolve the project to use based on configuration."""
        if self._project_id and not valid_key(self._project_id):
            raise RuntimeError(
                f'Invalid Project Key: "{self._project_id}". '
                "The expected characters are lowercase letters, numbers, and hyphens (-)."
            )

        project_key = self._project_id or DEFAULT_PROJECT_KEY
        found_project: Project | None = None

        try:
            found_project = self.api.get_project(
                project_identifier=project_key,
                workspace_id=self.workspace.id,
            )
        except RuntimeError as e:
            if "404: Project not found" not in str(e):
                raise

        if not found_project:
            found_project = self.api.create_project(
                name=self._project_id or DEFAULT_PROJECT_NAME,
                key=project_key,
                workspace_id=self.workspace.id,
            )

        self._project = found_project

    @staticmethod
    def extract_project_components(
        path: str | None,
    ) -> tuple[str | None, str | None, str | None]:
        """
        Extract organization, workspace, and project from a path string.

        The path can be in format: `org/workspace/project`, `workspace/project`, or `project`.

        Returns:
            Tuple of (organization, workspace, project). Missing components are None.
        """
        if not path:
            return (None, None, None)

        pattern = r"^(?:([\s\w-]+?)/)?(?:([\s\w-]+?)/)?([\s\w-]+?)$"
        match = re.match(pattern, path)

        if not match:
            raise RuntimeError(
                f"Invalid project path format: '{path}'. "
                "Expected: 'org/workspace/project', 'workspace/project', or 'project'."
            )

        groups = match.groups()
        present = [c for c in groups if c is not None]

        for component in present:
            if not valid_key(component):
                raise RuntimeError(
                    f'Invalid Key: "{component}". '
                    "Expected lowercase letters, numbers, and hyphens (-)."
                )

        if len(present) == 3:
            return groups[0], groups[1], groups[2]
        if len(present) == 2:
            return None, groups[1], groups[2]
        if len(present) == 1:
            return None, None, groups[2]
        return None, None, None
