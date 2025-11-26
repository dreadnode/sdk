import asyncio
import contextlib
import os
import random
import re
import sys
import typing as t
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse
from uuid import UUID

import coolname  # type: ignore [import-untyped]
import logfire
from fsspec.implementations.local import (  # type: ignore [import-untyped]
    LocalFileSystem,
)
from logfire._internal.exporters.remove_pending import RemovePendingSpansExporter
from opentelemetry import propagate
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from dreadnode.api.client import ApiClient
from dreadnode.api.models import Organization, Project, Workspace, WorkspaceFilter
from dreadnode.artifact.credential_manager import CredentialManager
from dreadnode.common_types import (
    INHERITED,
    AnyDict,
    Inherited,
    JsonValue,
)
from dreadnode.constants import (
    DEFAULT_LOCAL_STORAGE_DIR,
    DEFAULT_PROJECT_KEY,
    DEFAULT_PROJECT_NAME,
    DEFAULT_SERVER_URL,
    ENV_API_KEY,
    ENV_API_TOKEN,
    ENV_CONSOLE,
    ENV_LOCAL_DIR,
    ENV_ORGANIZATION,
    ENV_PROFILE,
    ENV_PROJECT,
    ENV_SERVER,
    ENV_SERVER_URL,
    ENV_WORKSPACE,
)
from dreadnode.error import AssertionFailedError
from dreadnode.exporter import CustomOTLPSpanExporter
from dreadnode.logging_ import console as logging_console
from dreadnode.metric import (
    Metric,
    MetricAggMode,
    MetricDict,
    MetricsLike,
    T,
)
from dreadnode.scorers import Scorer, ScorerCallable
from dreadnode.scorers.base import ScorersLike
from dreadnode.task import P, R, ScoredTaskDecorator, Task, TaskDecorator
from dreadnode.tracing.exporters import (
    FileExportConfig,
    FileMetricReader,
    FileSpanExporter,
)
from dreadnode.tracing.span import (
    RunContext,
    RunSpan,
    Span,
    TaskSpan,
    current_run_span,
    current_task_span,
)
from dreadnode.user_config import UserConfig
from dreadnode.util import (
    clean_str,
    create_key_from_name,
    handle_internal_errors,
    valid_key,
    warn_at_user_stacklevel,
)
from dreadnode.version import VERSION

if t.TYPE_CHECKING:
    from fsspec import AbstractFileSystem  # type: ignore [import-untyped]
    from opentelemetry.sdk.metrics.export import MetricReader
    from opentelemetry.sdk.trace import SpanProcessor
    from opentelemetry.trace import Tracer


ToObject = t.Literal["task-or-run", "run"]


class DreadnodeConfigWarning(UserWarning):
    """Warnings related to Dreadnode configuration."""


class DreadnodeUsageWarning(UserWarning):
    """Warnings related to Dreadnode usage."""


@dataclass
class Dreadnode:
    """
    The core Dreadnode SDK class.

    A default instance of this class is created and can be used directly with `dreadnode.*`.

    Otherwise, you can create your own instance and configure it with `configure()`.
    """

    server: str | None
    token: str | None
    local_dir: str | Path | t.Literal[False]
    organization: str | UUID | None
    workspace: str | UUID | None
    project: str | None
    service_name: str | None
    service_version: str | None
    console: logfire.ConsoleOptions | bool
    send_to_logfire: bool | t.Literal["if-token-present"]
    otel_scope: str

    def __init__(
        self,
        *,
        server: str | None = None,
        token: str | None = None,
        local_dir: str | Path | t.Literal[False] = False,
        organization: str | UUID | None = None,
        workspace: str | UUID | None = None,
        project: str | None = None,
        service_name: str | None = None,
        service_version: str | None = None,
        console: logfire.ConsoleOptions | bool = False,
        send_to_logfire: bool | t.Literal["if-token-present"] = False,
        otel_scope: str = "dreadnode",
    ) -> None:
        self.server = server
        self.token = token
        self.local_dir = local_dir
        self.organization = organization
        self.workspace = workspace
        self.project = project
        self.service_name = service_name
        self.service_version = service_version
        self.console = console
        self.send_to_logfire = send_to_logfire
        self.otel_scope = otel_scope

        self._api: ApiClient | None = None
        self._credential_manager: CredentialManager | None = None
        self._logfire = logfire.DEFAULT_LOGFIRE_INSTANCE
        self._logfire.config.ignore_no_config = True

        self._organization: Organization
        self._workspace: Workspace
        self._project: Project

        self._fs: AbstractFileSystem = LocalFileSystem(auto_mkdir=True)
        self._fs_prefix: str = f"{DEFAULT_LOCAL_STORAGE_DIR}/storage/"

        self._initialized = False

    def _get_profile_server(self, profile: str | None = None) -> str | None:
        """
        Get the server URL from the user config for a given profile.

        Args:
            profile: The profile name to use. If not provided, it will use the
                `DREADNODE_PROFILE` environment variable or the active profile.

        Returns:
            The server URL, or None if not found.
        """
        with contextlib.suppress(Exception):
            user_config = UserConfig.read()
            profile = profile or os.environ.get(ENV_PROFILE)
            server_config = user_config.get_server_config(profile)
            return server_config.url

        # Silently fail if profile config is not available or invalid
        return None

    def _get_profile_api_key(self, profile: str | None = None) -> str | None:
        """
        Get the API key from the user config for a given profile.

        Args:
            profile: The profile name to use. If not provided, it will use the
                `DREADNODE_PROFILE` environment variable or the active profile.

        Returns:
            The API key, or None if not found.
        """
        with contextlib.suppress(Exception):
            user_config = UserConfig.read()
            profile = profile or os.environ.get(ENV_PROFILE)
            server_config = user_config.get_server_config(profile)
            return server_config.api_key

        # Silently fail if profile config is not available or invalid
        return None

    def _resolve_organization(self) -> None:
        """
        Resolve the organization to use based on configuration.

        It will try to find the organization by ID or name if provided.
        If not, it will list organizations and use the only one available.

        Raises:
            RuntimeError: If the API client is not initialized, the organization
                is not found, or the user belongs to multiple organizations without
                specifying one.
        """
        if self._api is None:
            raise RuntimeError("API client is not initialized.")

        with contextlib.suppress(ValueError):
            self.organization = UUID(
                str(self.organization)
            )  # Now, it's a UUID if possible, else str (name/slug)

        if isinstance(self.organization, str) and not valid_key(self.organization):
            raise RuntimeError(
                f'Invalid Organization Key: "{self.organization}". The expected characters are lowercase letters, numbers, and hyphens (-).\n\nYou can get the keys for your organization using the CLI or the web interface.',
            )

        if self.organization:
            self._organization = self._api.get_organization(self.organization)
            if not self._organization:
                raise RuntimeError(f"Organization '{self.organization}' not found.")

        else:
            organizations = self._api.list_organizations()

            if not organizations:
                raise RuntimeError(
                    f"You are not part of any organizations on {self.server}. You will not be able to use Strikes.",
                )

            if len(organizations) > 1:
                # We should not presume to choose an organization
                org_list = "\t\n".join([f"- {o.name}" for o in organizations])
                raise RuntimeError(
                    f"You are part of multiple organizations. Please specify an organization from:\n{org_list}"
                )
            self._organization = organizations[0]

    def _create_workspace(self, key: str) -> Workspace:
        """
        Create a new workspace.

        Args:
            name: The name of the workspace to create.

        Returns:
            The created Workspace object.

        Raises:
            RuntimeError: If the API client is not initialized or if the user
                does not have permission to create a workspace.
        """
        if self._api is None:
            raise RuntimeError("API client is not initialized.")

        try:
            logging_console.print(
                f"[yellow]WARNING: This workspace was not found. Creating a new workspace '{key}'...[/]"
            )
            key = create_key_from_name(key)
            return self._api.create_workspace(
                name=key, key=key, organization_id=self._organization.id
            )
        except RuntimeError as e:
            if "403: Forbidden" in str(e):
                raise RuntimeError(
                    "You do not have permission to create the specified workspace for this organization. (You must be a Contributor to create new workspaces)."
                ) from e
            raise

    def _resolve_workspace(self) -> None:
        """
        Resolve the workspace to use based on configuration.

        If a workspace is specified by name and doesn't exist, it will be created.
        If no workspace is specified, it will look for a default workspace or
        create one named 'default'.

        Raises:
            RuntimeError: If the API client is not initialized, a specified
                workspace ID is not found, or if it fails to resolve or create a
                workspace.
        """
        if self._api is None:
            raise RuntimeError("API client is not initialized.")

        with contextlib.suppress(ValueError):
            self.workspace = UUID(
                str(self.workspace)
            )  # Now, it's a UUID if possible, else str (name/slug)

        if isinstance(self.workspace, str) and not valid_key(self.workspace):
            raise RuntimeError(
                f'Invalid Workspace Key: "{self.workspace}". The expected characters are lowercase letters, numbers, and hyphens (-).\n\nYou can get the keys for your workspace using the CLI or the web interface.',
            )

        found_workspace: Workspace | None = None
        if self.workspace:
            try:
                found_workspace = self._api.get_workspace(
                    self.workspace, org_id=self._organization.id
                )
            except RuntimeError as e:
                if "404: Workspace not found" in str(e):
                    pass  # do nothing, we'll create it below
                else:
                    raise

            if not found_workspace and isinstance(self.workspace, UUID):
                raise RuntimeError(f"Workspace with ID '{self.workspace}' not found.")

            if not found_workspace and isinstance(self.workspace, str):  # specified by name/slug
                # create the workspace (must be an org contributor)
                found_workspace = self._create_workspace(key=self.workspace)

        else:  # the user provided no workspace, attempt to find a default one
            workspaces = self._api.list_workspaces(
                filters=WorkspaceFilter(
                    org_id=self._organization.id if self._organization else None
                )
            )
            default_workspace = next((ws for ws in workspaces if ws.is_default is True), None)
            if default_workspace:
                found_workspace = default_workspace
            else:
                raise RuntimeError(
                    "No default workspace found. Please specify a workspace which you have contributor access to."
                )

        if not found_workspace:
            raise RuntimeError("Failed to resolve or create a workspace.")

        self._workspace = found_workspace

    def _resolve_project(self) -> None:
        """
        Resolve the project to use based on configuration.

        If a project is specified by key and doesn't exist, it will be created.
        If no project is specified, it will use or create one with key 'default'.

        Raises:
            RuntimeError: If the API client is not initialized.
        """
        if self._api is None:
            raise RuntimeError("API client is not initialized.")

        if self.project and not valid_key(self.project):
            raise RuntimeError(
                f'Invalid Project Key: "{self.project}". The expected characters are lowercase letters, numbers, and hyphens (-).\n\nYou can get the keys for your project using the CLI or the web interface.',
            )

        # fetch the project
        found_project: Project | None = None
        try:
            found_project = self._api.get_project(
                project_identifier=self.project or DEFAULT_PROJECT_KEY,
                workspace_id=self._workspace.id,
            )
        except RuntimeError as e:
            if "404: Project not found" in str(e):
                pass  # do nothing, we'll create it below
            else:
                raise

        if not found_project:
            # create it in the workspace
            found_project = self._api.create_project(
                name=self.project or DEFAULT_PROJECT_NAME,
                key=self.project or DEFAULT_PROJECT_KEY,
                workspace_id=self._workspace.id,
            )
        # This is what's used in all of the Traces/Spans/Runs
        self._project = found_project
        self.project = str(self._project.id)

    def _resolve_rbac(self) -> None:
        """
        Resolve organization, workspace, and project for RBAC.

        This is a convenience method that calls the individual resolution methods.

        Raises:
            RuntimeError: If the API client is not initialized.
        """
        if self._api is None:
            raise RuntimeError("API client is not initialized.")
        self._resolve_organization()
        self._resolve_workspace()
        self._resolve_project()

    def _log_configuration(self, config_source: str, active_profile: str | t.Any | None) -> None:
        """
        Log the current Dreadnode configuration to the console.

        Args:
            config_source: A string indicating where the configuration came from.
            active_profile: The name of the active profile, if any.
        """
        logging_console.print(f"Dreadnode Configuration: (from {config_source})")

        if self.server or self.token:
            destination = self.server or DEFAULT_SERVER_URL or "local storage"
            logging_console.print(f" Server: [orange_red1]{destination}[/]")
        elif self.local_dir:
            logging_console.print(
                f"Local directory: [orange_red1]{self.local_dir}[/] ({config_source})"
            )

        # Warn the user if the profile didn't resolve
        elif active_profile and not (self.server or self.token):
            logging_console.print(
                f":exclamation: Dreadnode profile [orange_red1]{active_profile}[/] appears invalid."
            )
        logging_console.print(f" Organization: [green]{self._organization.name}[/]")
        logging_console.print(f" Workspace: [green]{self._workspace.name}[/]")
        logging_console.print(f" Project: [green]{self._project.name}[/]")

    @staticmethod
    def _extract_project_components(path: str | None) -> tuple[str | None, str | None, str]:
        """
        Extract organization, workspace, and project from a path string.

        The path can be in the format `org/workspace/project`, `workspace/project`,
        or `project`.

        Args:
            path: The path string to parse.

        Returns:
            A tuple containing (organization, workspace, project). Components that
            are not present will be None.
        """
        if not path:
            return (None, None, "")

        pattern = r"^(?:([\s\w-]+?)/)?(?:([\s\w-]+?)/)?([\s\w-]+?)$"
        match = re.match(pattern, path)

        if not match:
            raise RuntimeError(
                f"Invalid project path format: '{path}'.\n\nExpected formats are 'org/workspace/project', 'workspace/project', or 'project'. Where each component is the key for that entity.'"
            )

        # The groups are: (Org, Workspace, Project)
        groups = match.groups()

        present_components = [c for c in groups if c is not None]

        # validate each component
        for component in present_components:
            if not valid_key(component):
                raise RuntimeError(
                    f'Invalid Key: "{component}". The expected characters are lowercase letters, numbers, and hyphens (-).\n\nYou can get the keys for your organization, workspace, and project using the CLI or the web interface.',
                )

        if len(present_components) == 3:
            org, workspace, project = groups
        elif len(present_components) == 2:
            org = None
            workspace, project = groups[1], groups[2]
        elif len(present_components) == 1:
            org = None
            workspace = None
            project = groups[2]
        else:
            raise RuntimeError("Regex matched, but component count is unexpected.")

        return (org, workspace, project)

    def get_current_run(self) -> RunSpan | None:
        return current_run_span.get()

    def get_current_task(self) -> TaskSpan[t.Any] | None:
        return current_task_span.get()

    def configure(
        self,
        *,
        server: str | None = None,
        token: str | None = None,
        profile: str | None = None,
        local_dir: str | Path | t.Literal[False] = False,
        organization: str | UUID | None = None,
        workspace: str | UUID | None = None,
        project: str | None = None,
        service_name: str | None = None,
        service_version: str | None = None,
        console: logfire.ConsoleOptions | bool | None = None,
        send_to_logfire: bool | t.Literal["if-token-present"] = False,
        otel_scope: str = "dreadnode",
    ) -> None:
        """
        Configure the Dreadnode SDK and call `initialize()`.

        This method should always be called before using the SDK.

        If `server` and `token` are not provided, the SDK will look for them
        in the following order:

        1. Environment variables:
           - `DREADNODE_SERVER_URL` or `DREADNODE_SERVER`
           - `DREADNODE_API_TOKEN` or `DREADNODE_API_KEY`
           - `DREADNODE_ORGANIZATION`
           - `DREADNODE_WORKSPACE`
           - `DREADNODE_PROJECT`

        2. Dreadnode profile (from `dreadnode login`)
           - Uses `profile` parameter if provided
           - Falls back to `DREADNODE_PROFILE` environment variable
           - Defaults to active profile

        Args:
            server: The Dreadnode server URL.
            token: The Dreadnode API token.
            profile: The Dreadnode profile name to use (only used if env vars are not set).
            local_dir: The local directory to store data in.
            organization: The default organization name or ID to use.
            workspace: The default workspace name or ID to use.
            project: The default project name to associate all runs with. This can also be in the format `org/workspace/project` using the keys.
            service_name: The service name to use for OpenTelemetry.
            service_version: The service version to use for OpenTelemetry.
            console: Log span information to the console (`DREADNODE_CONSOLE` or the default is True).
            send_to_logfire: Send data to Logfire.
            otel_scope: The OpenTelemetry scope name.
        """

        self._initialized = False

        # Skip during testing
        if "pytest" in sys.modules:
            self._initialized = True
            return

        # Determine configuration source and active profile for logging
        config_source = "explicit parameters"
        active_profile = None

        if not server or not token:
            # Check environment variables first
            env_server = os.environ.get(ENV_SERVER_URL) or os.environ.get(ENV_SERVER)
            env_token = os.environ.get(ENV_API_TOKEN) or os.environ.get(ENV_API_KEY)

            if env_server or env_token:
                config_source = "environment vars"
            else:
                # Fall back to profile
                config_source = "profile"
                with contextlib.suppress(Exception):
                    user_config = UserConfig.read()
                    profile_name = profile or os.environ.get(ENV_PROFILE)
                    active_profile = profile_name or user_config.active_profile_name

                    if active_profile:
                        config_source = f"profile: {active_profile}"

        self.server = (
            server
            or os.environ.get(ENV_SERVER_URL)
            or os.environ.get(ENV_SERVER)
            or self._get_profile_server(profile)
        )
        self.token = (
            token
            or os.environ.get(ENV_API_TOKEN)
            or os.environ.get(ENV_API_KEY)
            or self._get_profile_api_key(profile)
        )

        if local_dir is False and ENV_LOCAL_DIR in os.environ:
            env_local_dir = os.environ.get(ENV_LOCAL_DIR)
            if env_local_dir:
                self.local_dir = Path(env_local_dir)
            else:
                self.local_dir = False
        else:
            self.local_dir = local_dir

        _org, _workspace, _project = self._extract_project_components(project)
        self.organization = _org or organization or os.environ.get(ENV_ORGANIZATION)
        self.workspace = _workspace or workspace or os.environ.get(ENV_WORKSPACE)
        self.project = _project or project or os.environ.get(ENV_PROJECT)

        self.service_name = service_name
        self.service_version = service_version
        self.console = (
            console
            if console is not None
            else os.environ.get(ENV_CONSOLE, "false").lower()
            in [
                "true",
                "1",
                "yes",
            ]
        )
        self.send_to_logfire = send_to_logfire
        self.otel_scope = otel_scope

        self.initialize()

        self._log_configuration(config_source, active_profile)

    def initialize(self) -> None:
        """
        Initialize the Dreadnode SDK.

        This method is called automatically when you call `configure()`.
        """

        if self._initialized:
            return

        span_processors: list[SpanProcessor] = []
        metric_readers: list[MetricReader] = []

        self.server = self.server or (DEFAULT_SERVER_URL if self.token else None)
        if not (self.server or self.token or self.local_dir):
            warn_at_user_stacklevel(
                "Your current configuration won't persist run data anywhere. "
                "Login with `dreadnode login` to set up a server and token, "
                "Use `dreadnode.configure(server=..., token=...)`, `dreadnode.configure(profile=...)`, "
                f"or use environment variables ({ENV_SERVER_URL}, {ENV_API_TOKEN}, {ENV_LOCAL_DIR}).",
                category=DreadnodeConfigWarning,
            )

        if self.local_dir:
            config = FileExportConfig(
                base_path=self.local_dir,
                prefix=self.project + "-" if self.project else "",
            )
            span_processors.append(BatchSpanProcessor(FileSpanExporter(config)))
            metric_readers.append(FileMetricReader(config))

        if self.token and self.server:
            try:
                parsed_url = urlparse(self.server)
                if not parsed_url.scheme:
                    netloc = parsed_url.path.split("/")[0]
                    path = "/".join(parsed_url.path.split("/")[1:])
                    parsed_new = parsed_url._replace(
                        scheme="https", netloc=netloc, path=f"/{path}" if path else ""
                    )
                    self.server = urlunparse(parsed_new)

                self._api = ApiClient(self.server, api_key=self.token)
                self._resolve_rbac()
            except Exception as e:
                raise RuntimeError(
                    f"Failed to connect to {self.server}: {e}",
                ) from e

            headers = {"X-Api-Key": self.token}
            endpoint = "/api/otel/traces"
            span_processors.append(
                BatchSpanProcessor(
                    RemovePendingSpansExporter(  # This will tell Logfire to emit pending spans to us as well
                        CustomOTLPSpanExporter(
                            endpoint=urljoin(self.server, endpoint),
                            headers=headers,
                            compression=Compression.Gzip,
                        ),
                    ),
                ),
            )
            # TODO(nick): Metrics
            # https://linear.app/dreadnode/issue/ENG-1310/sdk-add-metrics-exports
            # metric_readers.append(
            #     PeriodicExportingMetricReader(
            #         OTLPMetricExporter(
            #             endpoint=urljoin(self.server, "/v1/metrics"),
            #             headers=headers,
            #             compression=Compression.Gzip,
            #             # preferred_temporality
            #         )
            #     )
            # )
            if self._api is not None:
                api = self._api
                self._credential_manager = CredentialManager(
                    credential_fetcher=lambda: api.get_user_data_credentials(
                        self._organization.id, self._workspace.id
                    )
                )

                self._credential_manager.initialize()

                self._fs = self._credential_manager.get_filesystem()
                self._fs_prefix = self._credential_manager.get_prefix()

        self._logfire = logfire.configure(
            local=not self.is_default,
            send_to_logfire=self.send_to_logfire,
            additional_span_processors=span_processors,
            metrics=logfire.MetricsOptions(additional_readers=metric_readers),
            service_name=self.service_name,
            service_version=self.service_version,
            console=logfire.ConsoleOptions() if self.console is True else self.console,
            scrubbing=False,
            inspect_arguments=False,
            distributed_tracing=False,
        )
        self._logfire.config.ignore_no_config = True

        self._initialized = True

    @property
    def is_default(self) -> bool:
        return self is DEFAULT_INSTANCE

    def api(self, *, server: str | None = None, token: str | None = None) -> ApiClient:
        """
        Get an API client based on the current configuration or the provided server and token.

        If the server and token are not provided, the method will use the current configuration
        and `configure()` needs to be called first.

        Args:
            server: The server URL to use for the API client.
            token: The API token to use for authentication.

        Returns:
            An ApiClient instance.
        """
        if server is not None and token is not None:
            return ApiClient(server, api_key=token)

        if not self._initialized:
            raise RuntimeError("Call .configure() before accessing the API")

        if self._api is None:
            raise RuntimeError("API is not available without a server configuration")

        return self._api

    def _get_tracer(self, *, is_span_tracer: bool = True) -> "Tracer":
        """
        Get an OpenTelemetry Tracer instance.

        Args:
            is_span_tracer: Whether the tracer is for creating spans.

        Returns:
            An OpenTelemetry Tracer.
        """
        return self._logfire._tracer_provider.get_tracer(  # noqa: SLF001
            self.otel_scope,
            VERSION,
            is_span_tracer=is_span_tracer,
        )

    @handle_internal_errors()
    def shutdown(self) -> None:
        """
        Shutdown any associate OpenTelemetry components and flush any pending spans.

        It is not required to call this method, as the SDK will automatically
        flush and shutdown when the process exits.

        However, if you want to ensure that all spans are flushed before
        exiting, you can call this method manually.
        """
        if not self._initialized:
            return

        self._logfire.shutdown()

    def span(
        self,
        name: str,
        *,
        tags: t.Sequence[str] | None = None,
        attributes: AnyDict | None = None,
    ) -> Span:
        """
        Create a new OpenTelemety span.

        Spans are more lightweight than tasks, but still let you track
        work being performed and view it in the UI. You cannot
        log parameters, inputs, or outputs to spans.

        Example:
            ```
            with dreadnode.span("my_span") as span:
                # do some work here
                pass
            ```

        Args:
            name: The name of the span.
            tags: A list of tags to attach to the span.
            attributes: A dictionary of attributes to attach to the span.

        Returns:
            A Span object.
        """
        return Span(
            name=name,
            attributes=attributes,
            tracer=self._get_tracer(),
            tags=tags,
        )

    @t.overload
    def task(
        self,
        func: t.Callable[P, t.Awaitable[R]],
        /,
    ) -> Task[P, R]: ...

    @t.overload
    def task(
        self,
        func: t.Callable[P, R],
        /,
    ) -> Task[P, R]: ...

    @t.overload
    def task(
        self,
        func: None = None,
        /,
        *,
        scorers: None = None,
        assert_scores: None = None,
        name: str | None = None,
        label: str | None = None,
        log_inputs: t.Sequence[str] | bool | Inherited = INHERITED,
        log_output: bool | Inherited = INHERITED,
        log_execution_metrics: bool = False,
        tags: t.Sequence[str] | None = None,
        attributes: AnyDict | None = None,
        entrypoint: bool = False,
    ) -> TaskDecorator: ...

    @t.overload
    def task(
        self,
        func: None = None,
        /,
        *,
        scorers: ScorersLike[R],
        assert_scores: list[str] | t.Literal[True] | None = None,
        name: str | None = None,
        label: str | None = None,
        log_inputs: t.Sequence[str] | bool | Inherited = INHERITED,
        log_output: bool | Inherited = INHERITED,
        log_execution_metrics: bool = False,
        tags: t.Sequence[str] | None = None,
        attributes: AnyDict | None = None,
        entrypoint: bool = False,
    ) -> ScoredTaskDecorator[R]: ...

    def task(
        self,
        func: t.Callable[P, t.Awaitable[R]] | t.Callable[P, R] | None = None,
        /,
        *,
        scorers: ScorersLike[t.Any] | None = None,
        assert_scores: list[str] | t.Literal[True] | None = None,
        name: str | None = None,
        label: str | None = None,
        log_inputs: t.Sequence[str] | bool | Inherited = INHERITED,
        log_output: bool | Inherited = INHERITED,
        log_execution_metrics: bool = False,
        tags: t.Sequence[str] | None = None,
        attributes: AnyDict | None = None,
        entrypoint: bool = False,
    ) -> TaskDecorator | ScoredTaskDecorator[R] | Task[P, R]:
        """
        Create a new task from a function.

        Example:
            ```
            @dreadnode.task
            async def my_task(x: int) -> int:
                return x * 2

            await my_task(2)
            ```

        Args:
            scorers: A list of scorers to attach to the task. These will be called after every execution
                of the task and will be passed the task's output.
            assert_scores: A list of score names to ensure have truthy values, otherwise raise an AssertionFailedError.
            name: The name of the task.
            label: The label of the task - useful for filtering in the UI.
            log_inputs: Log all, or specific, incoming arguments to the function as inputs.
            log_output: Log the result of the function as an output.
            log_execution_metrics: Log execution metrics for the task, such as success rate and run count.
            tags: A list of tags to attach to the task span.
            attributes: A dictionary of attributes to attach to the task span.
            entrypoint: Indicate this task should be considered an entrypoint. All compatible arguments
                will be treated as configurable and a run will be created automatically when called if
                one is not already active.

        Returns:
            A new Task object.
        """

        # NOTE(nick): It would probably be cleaner to alias a `dn.entrypoint` decorator
        # that just wraps `dn.task(..., entrypoint=True)`, but the overloads create quite
        # a bit of duplicate code, so I'm leaving it like this for now.

        if isinstance(func, Task):
            return func

        def make_task(
            func: t.Callable[P, t.Awaitable[R]] | t.Callable[P, R],
        ) -> Task[P, R]:
            if isinstance(func, Task):
                return func.with_(
                    name=name,
                    scorers=scorers,  # type: ignore[arg-type]
                    assert_scores=assert_scores,
                    label=label,
                    log_inputs=log_inputs,
                    log_output=log_output,
                    log_execution_metrics=log_execution_metrics,
                    tags=tags,
                    attributes=attributes,
                    entrypoint=entrypoint,
                    append=True,
                )

            return Task(
                func=t.cast("t.Callable[P, R]", func),
                tracer=self._get_tracer(),
                name=name,
                label=label,
                scorers=scorers,
                assert_scores=assert_scores,
                log_inputs=log_inputs,
                log_output=log_output,
                log_execution_metrics=log_execution_metrics,
                tags=tags,
                attributes=attributes,
                entrypoint=entrypoint,
            )

        return (
            t.cast("TaskDecorator | ScoredTaskDecorator[R]", make_task)
            if func is None
            else make_task(func)
        )

    def task_span(
        self,
        name: str,
        *,
        label: str | None = None,
        tags: t.Sequence[str] | None = None,
        attributes: AnyDict | None = None,
        _tracer: "Tracer | None" = None,
    ) -> TaskSpan[t.Any]:
        """
        Create a task span without an explicit associated function.

        This is useful for creating tasks on the fly without having to
        define a function.

        Example:
            ```
            async with dreadnode.task_span("my_task") as task:
                # do some work here
                pass
            ```
        Args:
            name: The name of the task.
            label: The label of the task - useful for filtering in the UI.
            tags: A list of tags to attach to the task span.
            attributes: A dictionary of attributes to attach to the task span.

        Returns:
            A TaskSpan object.
        """
        run = current_run_span.get()
        label = clean_str(label or name)

        return TaskSpan(
            name=name,
            label=label,
            attributes=attributes,
            tags=tags,
            run_id=run.run_id if run else "",
            tracer=_tracer or self._get_tracer(),
        )

    @t.overload
    def scorer(
        self,
        func: None = None,
        /,
        *,
        name: str | None = None,
        attributes: AnyDict | None = None,
    ) -> t.Callable[[ScorerCallable[T]], Scorer[T]]: ...

    @t.overload
    def scorer(
        self,
        func: ScorerCallable[T],
        /,
    ) -> Scorer[T]: ...

    def scorer(
        self,
        func: ScorerCallable[T] | None = None,
        *,
        name: str | None = None,
        attributes: AnyDict | None = None,
    ) -> t.Callable[[ScorerCallable[T]], Scorer[T]] | Scorer[T]:
        """
        Make a scorer from a callable function.

        This is useful when you want to change the name of the scorer
        or add additional attributes to it.

        Example:
            ```
            @dreadnode.scorer
            async def my_scorer(x: int) -> float:
                return x * 2

            @dreadnode.task(scorers=[my_scorer])
            async def my_task(x: int) -> int:
                return x * 2

            await my_task(2)
            ```

        Args:
            name: The name of the scorer.
            attributes: A dictionary of attributes to attach to the scorer.

        Returns:
            A new Scorer object.
        """

        if isinstance(func, Scorer):
            return func

        def make_scorer(func: ScorerCallable[T]) -> Scorer[T]:
            if isinstance(func, Scorer):
                return func.with_(name=name, attributes=attributes)
            return Scorer(func, name=name, attributes=attributes)

        return make_scorer if func is None else make_scorer(func)

    async def score(
        self,
        object: T,
        scorers: ScorersLike[T],
        step: int | None = None,
        assert_scores: list[str] | t.Literal[True] | None = None,
    ) -> dict[str, list[Metric]]:
        """
        Score an object using all the provided scorers.

        Args:
            object: The object to score.
            scorers: A list of scorers to use for scoring the object.
            step: An optional step value to attach to all generated metrics.
            assert_scores: A list of score names to ensure have truthy values - otherwise raise an AssertionFailedError.

        Returns:
            A dictionary of metrics generated by the scorers.
        """
        if not self._initialized:
            self.configure()

        _scorers = Scorer.fit_many(scorers)
        _assert_scores = (
            [s.name for s in _scorers] if assert_scores is True else list(assert_scores or [])
        )

        metrics: dict[str, list[Metric]] = {}
        nested_metrics = await asyncio.gather(
            *[scorer.normalize_and_score(object) for scorer in _scorers]
        )
        for scorer, _metrics in zip(_scorers, nested_metrics, strict=True):
            for metric in _metrics:
                if step is not None:
                    metric.step = step
                metric_name = str(getattr(metric, "_scorer_name", scorer.name))
                metric_name = clean_str(metric_name)
                metrics.setdefault(metric_name, []).append(
                    self.log_metric(metric_name, metric, origin=scorer.bound_obj or object)
                )

        failed_assertions: dict[str, list[Metric]] = {}
        for name in _assert_scores:
            if (metric_list := metrics.get(name, [])) is None:
                for _metrics in metrics.values():
                    if getattr(_metrics[0], "_scorer_name", None) == name:
                        metric_list = _metrics
                        break

            if not any(m.value for m in metric_list):
                failed_assertions[name] = metric_list

        if failed_assertions:
            raise AssertionFailedError(
                f"{len(failed_assertions)} score assertion(s) failed: {list(failed_assertions.keys())}",
                failures=failed_assertions,
            )

        return metrics

    def run(
        self,
        name: str | None = None,
        *,
        tags: t.Sequence[str] | None = None,
        params: AnyDict | None = None,
        project: str | None = None,
        autolog: bool = True,
        name_prefix: str | None = None,
        attributes: AnyDict | None = None,
        _tracer: "Tracer | None" = None,
    ) -> RunSpan:
        """
        Create a new run.

        Runs are the main way to track work in Dreadnode. They are
        associated with a specific project and can have parameters,
        inputs, and outputs logged to them.

        You cannot create runs inside other runs.

        Example:
            ```
            with dreadnode.run("my_run"):
                # do some work here
                pass
            ```

        Args:
            name: The name of the run. If not provided, a random name will be generated.
            tags: A list of tags to attach to the run.
            params: A dictionary of parameters to attach to the run.
            project: The project name to associate the run with. If not provided,
                the project passed to `configure()` will be used, or the
                run will be associated with a default project.
            autolog: Automatically log task inputs, outputs, and execution metrics if otherwise unspecified.
            attributes: Additional attributes to attach to the run span.

        Returns:
            A RunSpan object that can be used as a context manager.
            The run will automatically be completed when the context manager exits.
        """
        if not self._initialized:
            self.configure()

        name_prefix = clean_str(name_prefix or coolname.generate_slug(2), replace_with="-")
        name = name or f"{name_prefix}-{random.randint(100, 999)}"  # noqa: S311 # nosec

        return RunSpan(
            name=name,
            project=project or self.project or "default",
            attributes=attributes,
            tracer=_tracer or self._get_tracer(),
            params=params,
            tags=tags,
            credential_manager=self._credential_manager,
            autolog=autolog,
        )

    @contextlib.contextmanager
    def task_and_run(
        self,
        name: str,
        *,
        project: str | None = None,
        tags: t.Sequence[str] | None = None,
        params: AnyDict | None = None,
        autolog: bool = True,
        inputs: AnyDict | None = None,
        label: str | None = None,
        _tracer: "Tracer | None" = None,
    ) -> t.Iterator[TaskSpan[t.Any]]:
        """
        Create a task span within a new run if one is not already active.
        """

        create_run = current_run_span.get() is None
        with contextlib.ExitStack() as stack:
            if create_run:
                stack.enter_context(
                    self.run(
                        name_prefix=name,
                        project=project,
                        tags=tags,
                        params=params,
                        autolog=autolog,
                        _tracer=_tracer,
                    )
                )
                self.log_inputs(**(inputs or {}), to="run")

            task_span = stack.enter_context(
                self.task_span(name, label=label, tags=tags, _tracer=_tracer)
            )
            self.log_inputs(**(inputs or {}))
            if not create_run:
                self.log_inputs(**(params or {}))

            yield task_span

    def get_run_context(self) -> RunContext:
        """
        Capture the current run context for transfer to another host, thread, or process.

        Use `continue_run()` to continue the run anywhere else.

        Returns:
            RunContext containing run state and trace propagation headers.

        Raises:
            RuntimeError: If called outside of an active run.
        """
        if (run := current_run_span.get()) is None:
            raise RuntimeError("get_run_context() must be called within a run")

        # Capture OpenTelemetry trace context
        trace_context: dict[str, str] = {}
        propagate.inject(trace_context)

        return {
            "run_id": run.run_id,
            "run_name": run.name,
            "project": run.project_id,
            "trace_context": trace_context,
        }

    def continue_run(self, run_context: RunContext) -> RunSpan:
        """
        Continue a run from captured context on a remote host.

        Args:
            run_context: The RunContext captured from get_run_context().

        Returns:
            A RunSpan object that can be used as a context manager.
        """
        if not self._initialized:
            self.configure()

        return RunSpan.from_context(
            context=run_context,
            tracer=self._get_tracer(),
            credential_manager=self._credential_manager,  # type: ignore[arg-type]
        )

    def tag(self, *tag: str, to: ToObject | t.Literal["both"] = "task-or-run") -> None:
        """
        Add one or many tags to the current task or run.

        Example:
            ```
            with dreadnode.run("my_run"):
                dreadnode.tag("my_tag")
            ```

        Args:
            tag: The tag to attach to the task or run.
            to: The target object to log the tag to. Can be "task-or-run" or "run".
                Defaults to "task-or-run". If "task-or-run", the tag will be logged
                to the current task or run, whichever is the nearest ancestor.
        """
        task = current_task_span.get()
        run = current_run_span.get()

        targets = [(task or run)] if to == "task-or-run" else [task, run] if to == "both" else [run]
        if not targets:
            warn_at_user_stacklevel(
                "tag() was called outside of a task or run.",
                category=DreadnodeUsageWarning,
            )
            return

        for target in [target for target in targets if target]:
            target.add_tags(tag)

    @handle_internal_errors()
    def push_update(self) -> None:
        """
        Push any pending run data to the server before run completion.

        This is useful for ensuring that the UI is up to date with the
        latest data. Data is automatically pushed periodically, but
        you can call this method to force a push.

        Example:
            ```
            with dreadnode.run("my_run"):
                dreadnode.log_params(...)
                dreadnode.log_metric(...)
                dreadnode.push_update()

                # do more work
        """
        if (run := current_run_span.get()) is None:
            warn_at_user_stacklevel(
                "push_update() was called outside of a run.",
                category=DreadnodeUsageWarning,
            )
            return

        run.push_update(force=True)

    @handle_internal_errors()
    def log_param(
        self,
        key: str,
        value: JsonValue,
    ) -> None:
        """
        Log a single parameter to the current run.

        Parameters are key-value pairs that are associated with the run
        and can be used to track configuration values, hyperparameters, or other
        metadata.

        Example:
            ```
            with dreadnode.run("my_run"):
                dreadnode.log_param("param_name", "param_value")
            ```

        Args:
            key: The name of the parameter.
            value: The value of the parameter.
        """
        self.log_params(**{key: value})

    @handle_internal_errors()
    def log_params(self, **params: JsonValue) -> None:
        """
        Log multiple parameters to the current run.

        Parameters are key-value pairs that are associated with the run
        and can be used to track configuration values, hyperparameters, or other
        metadata.

        Example:
            ```
            with dreadnode.run("my_run"):
                dreadnode.log_params(
                    param1="value1",
                    param2="value2"
                )
            ```

        Args:
            **params: The parameters to log. Each parameter is a key-value pair.
        """
        if (run := current_run_span.get()) is None:
            warn_at_user_stacklevel(
                "log_params() was called outside of a run.",
                category=DreadnodeUsageWarning,
            )
            return

        run.log_params(**params)

    @t.overload
    def log_metric(
        self,
        name: str,
        value: float | bool,  # noqa: FBT001
        *,
        step: int = 0,
        origin: t.Any | None = None,
        timestamp: datetime | None = None,
        mode: MetricAggMode | None = None,
        attributes: AnyDict | None = None,
        to: ToObject = "task-or-run",
    ) -> Metric:
        """
        Log a single metric to the current task or run.

        Metrics are some measurement or recorded value related to the task or run.
        They can be used to track performance, resource usage, or other quantitative data.

        Example:
            ```
            with dreadnode.run("my_run"):
                dreadnode.log_metric("metric_name", 42.0)
            ```

        Args:
            name: The name of the metric.
            value: The value of the metric.
            step: The step of the metric.
            origin: The origin of the metric - can be provided any object which was logged
                as an input or output anywhere in the run.
            timestamp: The timestamp of the metric - defaults to the current time.
            mode: The aggregation mode to use for the metric. Helpful when you want to let
                the library take care of translating your raw values into better representations.
                - direct: do not modify the value at all (default)
                - min: the lowest observed value reported for this metric
                - max: the highest observed value reported for this metric
                - avg: the average of all reported values for this metric
                - sum: the cumulative sum of all reported values for this metric
                - count: increment every time this metric is logged - disregard value
            attributes: A dictionary of additional attributes to attach to the metric.
            to: The target object to log the metric to. Can be "task-or-run" or "run".
                Defaults to "task-or-run". If "task-or-run", the metric will be logged
                to the current task or run, whichever is the nearest ancestor.

        Returns:
            The logged metric object.
        """

    @t.overload
    def log_metric(
        self,
        name: str,
        value: Metric,
        *,
        origin: t.Any | None = None,
        mode: MetricAggMode | None = None,
        to: ToObject = "task-or-run",
    ) -> Metric:
        """
        Log a single metric to the current task or run.

        Metrics are some measurement or recorded value related to the task or run.
        They can be used to track performance, resource usage, or other quantitative data.

        Example:
            ```
            with dreadnode.run("my_run"):
                dreadnode.log_metric("metric_name", 42.0)
            ```

        Args:
            name: The name of the metric.
            value: The metric object.
            origin: The origin of the metric - can be provided any object which was logged
                as an input or output anywhere in the run.
            mode: The aggregation mode to use for the metric. Helpful when you want to let
                the library take care of translating your raw values into better representations.
                - min: always report the lowest ovbserved value for this metric
                - max: always report the highest observed value for this metric
                - avg: report the average of all values for this metric
                - sum: report a rolling sum of all values for this metric
                - count: report the number of times this metric has been logged
            to: The target object to log the metric to. Can be "task-or-run" or "run".
                Defaults to "task-or-run". If "task-or-run", the metric will be logged
                to the current task or run, whichever is the nearest ancestor.

        Returns:
            The logged metric object.
        """

    @handle_internal_errors()
    def log_metric(
        self,
        name: str,
        value: float | bool | Metric,  # noqa: FBT001
        *,
        step: int = 0,
        origin: t.Any | None = None,
        timestamp: datetime | None = None,
        mode: MetricAggMode | None = None,
        attributes: AnyDict | None = None,
        to: ToObject = "task-or-run",
    ) -> Metric:
        """
        Log a single metric to the current task or run.

        Metrics are some measurement or recorded value related to the task or run.
        They can be used to track performance, resource usage, or other quantitative data.

        Examples:
            With a raw value:
            ```
            with dreadnode.run("my_run"):
                dreadnode.log_metric("accuracy", 0.95, step=10)
                dreadnode.log_metric("loss", 0.05, step=10, mode="min")
            ```

            With a Metric object:
            ```
            with dreadnode.run("my_run"):
                metric = Metric(0.95, step=10, timestamp=datetime.now(timezone.utc))
                dreadnode.log_metric("accuracy", metric)
            ```

        Args:
            name: The name of the metric.
            value: The value of the metric, either as a raw float/bool or a Metric object.
            step: The step of the metric.
            origin: The origin of the metric - can be provided any object which was logged
                as an input or output anywhere in the run.
            timestamp: The timestamp of the metric - defaults to the current time.
            mode: The aggregation mode to use for the metric. Helpful when you want to let
                the library take care of translating your raw values into better representations.
                - direct: do not modify the value at all (default)
                - min: the lowest observed value reported for this metric
                - max: the highest observed value reported for this metric
                - avg: the average of all reported values for this metric
                - sum: the cumulative sum of all reported values for this metric
                - count: increment every time this metric is logged - disregard value
            attributes: A dictionary of additional attributes to attach to the metric.
            to: The target object to log the metric to. Can be "task-or-run" or "run".
                Defaults to "task-or-run". If "task-or-run", the metric will be logged
                to the current task or run, whichever is the nearest ancestor.

        Returns:
            The logged metric object.
        """
        metric = (
            value
            if isinstance(value, Metric)
            else Metric(
                float(value),
                step,
                timestamp or datetime.now(timezone.utc),
                attributes or {},
            )
        )

        task = current_task_span.get()
        run = current_run_span.get()

        target = (task or run) if to == "task-or-run" else run
        if target is None:
            warn_at_user_stacklevel(
                "log_metric() was called outside of a task or run.",
                category=DreadnodeUsageWarning,
            )
            return metric

        return target.log_metric(name, metric, origin=origin, mode=mode)

    @t.overload
    def log_metrics(
        self,
        metrics: dict[str, float | bool],
        *,
        step: int = 0,
        timestamp: datetime | None = None,
        mode: MetricAggMode | None = None,
        attributes: AnyDict | None = None,
        origin: t.Any | None = None,
        to: ToObject = "task-or-run",
    ) -> list[Metric]:
        """
        Log multiple metrics from a dictionary of name/value pairs.

        Examples:
            ```
            dreadnode.log_metrics(
                {
                    "accuracy": 0.95,
                    "loss": 0.05,
                    "f1_score": 0.92
                },
                step=10
            )
            ```

        Args:
            metrics: Dictionary of name/value pairs to log as metrics.
            step: Step value for all metrics.
            timestamp: Timestamp for all metrics.
            mode: Aggregation mode for all metrics.
            attributes: Attributes for all metrics.
            to: The target object to log metrics to. Can be "task-or-run" or "run".
                Defaults to "task-or-run". If "task-or-run", the metrics will be logged
                to the current task or run, whichever is the nearest ancestor.

        Returns:
            List of logged Metric objects.
        """

    @t.overload
    def log_metrics(
        self,
        metrics: list[MetricDict],
        *,
        step: int = 0,
        timestamp: datetime | None = None,
        mode: MetricAggMode | None = None,
        attributes: AnyDict | None = None,
        origin: t.Any | None = None,
        to: ToObject = "task-or-run",
    ) -> list[Metric]:
        """
        Log multiple metrics from a list of metric configurations.

        Example:
            ```
            dreadnode.log_metrics(
                [
                    {"name": "accuracy", "value": 0.95},
                    {"name": "loss", "value": 0.05, "mode": "min"}
                ],
                step=10
            )
            ```

        Args:
            metrics: List of metric configurations to log.
            step: Default step value for metrics if not supplied.
            timestamp: Default timestamp for metrics if not supplied.
            mode: Default aggregation mode for metrics if not supplied.
            attributes: Default attributes for metrics if not supplied.
            to: The target object to log metrics to. Can be "task-or-run" or "run".
                Defaults to "task-or-run". If "task-or-run", the metrics will be logged
                to the current task or run, whichever is the nearest ancestor.

        Returns:
            List of logged Metric objects.
        """

    @handle_internal_errors()
    def log_metrics(
        self,
        metrics: MetricsLike,
        *,
        step: int = 0,
        timestamp: datetime | None = None,
        mode: MetricAggMode | None = None,
        attributes: AnyDict | None = None,
        origin: t.Any | None = None,
        to: ToObject = "task-or-run",
    ) -> list[Metric]:
        """
        Log multiple metrics to the current task or run.

        Examples:
            Log metrics from a dictionary:
            ```
            dreadnode.log_metrics(
                {
                    "accuracy": 0.95,
                    "loss": 0.05,
                    "f1_score": 0.92
                },
                step=10
            )
            ```

            Log metrics from a list of MetricDicts:
            ```
            dreadnode.log_metrics(
                [
                    {"name": "accuracy", "value": 0.95},
                    {"name": "loss", "value": 0.05, "mode": "min"}
                ],
                step=10
            )
            ```

        Args:
            metrics: Either a dictionary of name/value pairs or a list of MetricDicts to log.
            step: Default step value for metrics if not supplied.
            timestamp: Default timestamp for metrics if not supplied.
            mode: Default aggregation mode for metrics if not supplied.
            attributes: Default attributes for metrics if not supplied.
            origin: The origin of the metrics - can be provided any object which was logged
            to: The target object to log metrics to. Can be "task-or-run" or "run".
                Defaults to "task-or-run". If "task-or-run", the metrics will be logged
                to the current task or run, whichever is the nearest ancestor.

        Returns:
            List of logged Metric objects.
        """

        task = current_task_span.get()
        run = current_run_span.get()

        target = (task or run) if to == "task-or-run" else run
        if target is None:
            warn_at_user_stacklevel(
                "log_metrics() was called outside of a task or run.",
                category=DreadnodeUsageWarning,
            )
            return []

        logged_metrics: list[Metric] = []

        # Dictionary of name/value pairs
        if isinstance(metrics, dict):
            logged_metrics = [
                target.log_metric(
                    name,
                    value,
                    step=step,
                    timestamp=timestamp,
                    mode=mode,
                    attributes=attributes,
                    origin=origin,
                )
                for name, value in metrics.items()
            ]

        # List of MetricDicts
        else:
            logged_metrics = [
                target.log_metric(
                    metric["name"],
                    metric["value"],
                    step=metric.get("step", step),
                    timestamp=metric.get("timestamp", timestamp),
                    mode=metric.get("mode", mode),
                    attributes=metric.get("attributes", attributes) or {},
                    origin=origin,
                )
                for metric in metrics
            ]

        return logged_metrics

    @handle_internal_errors()
    def log_artifact(
        self,
        local_uri: str | Path,
    ) -> None:
        """
        Log a file or directory artifact to the current run.

        This method uploads a local file or directory to the artifact storage associated with the run.

        Examples:
            Log a single file:
            ```
            with dreadnode.run("my_run"):
                # Save a file
                with open("results.json", "w") as f:
                    json.dump(results, f)

                # Log it as an artifact
                dreadnode.log_artifact("results.json")
            ```

            Log a directory:
            ```
            with dreadnode.run("my_run"):
                # Create a directory with model files
                os.makedirs("model_output", exist_ok=True)
                save_model("model_output/model.pkl")
                save_config("model_output/config.yaml")

                # Log the entire directory as an artifact
                dreadnode.log_artifact("model_output")
            ```

        Args:
            local_uri: The local path to the file to upload.
        """
        if (run := current_run_span.get()) is None:
            warn_at_user_stacklevel(
                "log_artifact() was called outside of a run.",
                category=DreadnodeUsageWarning,
            )
            return

        run.log_artifact(local_uri=local_uri)

    @handle_internal_errors()
    def log_input(
        self,
        name: str,
        value: t.Any,
        *,
        label: str | None = None,
        to: ToObject | t.Literal["both"] = "task-or-run",
        attributes: AnyDict | None = None,
    ) -> None:
        """
        Log a single input to the current task or run.

        Inputs can be any runtime object, which are serialized, stored, and tracked
        in the Dreadnode UI.

        Example:
            ```
            @dreadnode.task
            async def my_task(x: int) -> int:
                dreadnode.log_input("input_name", x)
                return x * 2

            with dreadnode.run("my_run"):
                dreadnode.log_input("input_name", some_dataframe)

                await my_task(2)
            ```
        """
        task = current_task_span.get()
        run = current_run_span.get()

        targets = [(task or run)] if to == "task-or-run" else [task, run] if to == "both" else [run]
        if not targets:
            warn_at_user_stacklevel(
                "log_input() was called outside of a task or run.",
                category=DreadnodeUsageWarning,
            )
            return

        for target in [target for target in targets if target]:
            target.log_input(name, value, label=label, attributes=attributes)

    @handle_internal_errors()
    def log_inputs(
        self,
        to: ToObject | t.Literal["both"] = "task-or-run",
        **inputs: t.Any,
    ) -> None:
        """
        Log multiple inputs to the current task or run.

        See `log_input()` for more details.
        """
        for name, value in inputs.items():
            self.log_input(name, value, to=to)

    @handle_internal_errors()
    def log_output(
        self,
        name: str,
        value: t.Any,
        *,
        label: str | None = None,
        to: ToObject | t.Literal["both"] = "task-or-run",
        attributes: AnyDict | None = None,
    ) -> None:
        """
        Log a single output to the current task or run.

        Outputs can be any runtime object, which are serialized, stored, and tracked
        in the Dreadnode UI.

        Example:
            ```
            @dreadnode.task
            async def my_task(x: int) -> int:
                result = x * 2
                dreadnode.log_output("result", x * 2)
                return result

            with dreadnode.run("my_run"):
                await my_task(2)

                dreadnode.log_output("other", 123)
            ```

        Args:
            name: The name of the output.
            value: The value of the output.
            label: An optional label for the output, useful for filtering in the UI.
            to: The target object to log the output to. Can be "task-or-run" or "run".
                Defaults to "task-or-run". If "task-or-run", the output will be logged
                to the current task or run, whichever is the nearest ancestor.
            attributes: Additional attributes to attach to the output.
        """
        task = current_task_span.get()
        run = current_run_span.get()

        targets = [(task or run)] if to == "task-or-run" else [task, run] if to == "both" else [run]
        if not targets:
            warn_at_user_stacklevel(
                "log_output() was called outside of a task or run.",
                category=DreadnodeUsageWarning,
            )
            return

        for target in [target for target in targets if target]:
            target.log_output(name, value, label=label, attributes=attributes)

    @handle_internal_errors()
    def log_outputs(
        self,
        to: ToObject | t.Literal["both"] = "task-or-run",
        **outputs: t.Any,
    ) -> None:
        """
        Log multiple outputs to the current task or run.

        See `log_output()` for more details.
        """
        for name, value in outputs.items():
            self.log_output(name, value, to=to)

    @handle_internal_errors()
    def log_sample(
        self,
        label: str,
        input: t.Any,
        output: t.Any,
        metrics: MetricsLike | None = None,
        *,
        step: int = 0,
    ) -> None:
        """
        Convenience method to log an input/output pair with metrics as a ephemeral task.

        This is useful for logging a single sample of input and output data
        along with any metrics that were computed during the process.
        """

        with self.task_span(name=label, label=label):
            self.log_input("input", input)
            self.log_output("output", output)
            self.link_objects(output, input)
            if metrics is not None:
                self.log_metrics(metrics, step=step, origin=output)

    @handle_internal_errors()
    def log_samples(
        self,
        name: str,
        samples: list[tuple[t.Any, t.Any] | tuple[t.Any, t.Any, MetricsLike]],
    ) -> None:
        """
        Log multiple input/output samples as ephemeral tasks.

        This is useful for logging a batch of input/output pairs with metrics
        in a single run.

        Example:
            ```
            dreadnode.log_samples(
                "my_samples",
                [
                    (input1, output1, {"accuracy": 0.95}),
                    (input2, output2, {"accuracy": 0.90}),
                ]
            )
            ```

        Args:
            name: The name of the task to create for each sample.
            samples: A list of tuples containing (input, output, metrics [optional]).
        """
        for sample in samples:
            metrics: MetricsLike | None = None
            if len(sample) == 3:
                input_data, output_data, metrics = sample
            elif len(sample) == 2:
                input_data, output_data = sample
            else:
                raise ValueError(
                    "Each sample must be a tuple of (input, output) or (input, output, metrics)",
                )

            # Log each sample as an ephemeral task
            self.log_sample(name, input_data, output_data, metrics=metrics)

    @handle_internal_errors()
    def link_objects(
        self,
        origin: t.Any,
        link: t.Any,
        attributes: AnyDict | None = None,
    ) -> None:
        """
        Associate two runtime objects with each other.

        This is useful for linking any two objects which are related to
        each other, such as a model and its training data, or an input
        prompt and the resulting output.

        Example:
            ```
            with dreadnode.run("my_run"):
                model = SomeModel()
                data = SomeData()

                dreadnode.link_objects(model, data)
            ```

        Args:
            origin: The origin object to link from.
            link: The linked object to link to.
            attributes: Additional attributes to attach to the link.
        """
        if (run := current_run_span.get()) is None:
            warn_at_user_stacklevel(
                "link_objects() was called outside of a run.",
                category=DreadnodeUsageWarning,
            )
            return

        origin_hash = run.log_object(origin)
        link_hash = run.log_object(link)
        run.link_objects(origin_hash, link_hash, attributes=attributes)


DEFAULT_INSTANCE = Dreadnode()
