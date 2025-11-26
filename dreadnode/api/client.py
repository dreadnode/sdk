import io
import json
import shutil
import time
import typing as t
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from uuid import UUID

import httpx
from loguru import logger
from pydantic import BaseModel
from ulid import ULID

from dreadnode.api.models import (
    AccessRefreshTokenResponse,
    ContainerRegistryCredentials,
    DeviceCodeResponse,
    ExportFormat,
    GithubTokenResponse,
    MetricAggregationType,
    Organization,
    PaginatedWorkspaces,
    Project,
    RawRun,
    RawTask,
    RegistryImageDetails,
    Run,
    RunSummary,
    StatusFilter,
    Task,
    TaskTree,
    TimeAggregationType,
    TimeAxisType,
    TraceSpan,
    TraceTree,
    UserDataCredentials,
    UserResponse,
    Workspace,
    WorkspaceFilter,
)
from dreadnode.api.util import (
    convert_flat_tasks_to_tree,
    convert_flat_trace_to_tree,
    process_run,
    process_task,
)
from dreadnode.constants import (
    DEFAULT_MAX_POLL_TIME,
    DEFAULT_POLL_INTERVAL,
)
from dreadnode.version import VERSION

# NOTE(nick): Don't love the repeated `pandas` imports here,
# but this class is pretty central and I'd like to avoid
# pandas imports early as it's generally not needed

if t.TYPE_CHECKING:
    import pandas as pd

ModelT = t.TypeVar("ModelT", bound=BaseModel)


class ApiClient:
    """
    Client for the Dreadnode API.

    This class provides methods to interact with the Dreadnode API, including
    retrieving projects, runs, tasks, and exporting data.
    """

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        cookies: dict[str, str] | None = None,
        debug: bool = False,
        timeout: int = 30,
    ):
        """
        Initializes the API client.

        Args:
            base_url: The base URL of the Dreadnode API.
            api_key: The API key for authentication.
            cookies: A dictionary of cookies to include in requests.
            debug: Whether to enable debug logging. Defaults to False.
            timeout: The timeout for HTTP requests in seconds.
        """
        self._base_url = base_url.rstrip("/")
        if not self._base_url.endswith("/api"):
            self._base_url += "/api"

        _cookies = httpx.Cookies()
        cookie_domain = urlparse(base_url).hostname
        if cookie_domain is None:
            raise ValueError(f"Invalid URL: {base_url}")

        if cookie_domain == "localhost":
            cookie_domain = "localhost.local"

        for key, value in (cookies or {}).items():
            _cookies.set(key, value, domain=cookie_domain)

        headers = {
            "User-Agent": f"dreadnode-sdk/{VERSION}",
            "Accept": "application/json",
        }

        if api_key:
            headers["X-API-Key"] = api_key

        self._client = httpx.Client(
            headers=headers,
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout, connect=5),
            cookies=_cookies,
        )

        if debug:
            self._client.event_hooks["request"].append(self._log_request)
            self._client.event_hooks["response"].append(self._log_response)

    def _log_request(self, request: httpx.Request) -> None:
        """
        Logs HTTP requests if debug mode is enabled.

        Args:
            request (httpx.Request): The HTTP request object.
        """

        logger.debug("-------------------------------------------")
        logger.debug("%s %s", request.method, request.url)
        logger.debug("Headers: %s", request.headers)
        logger.debug("Content: %s", request.content)
        logger.debug("-------------------------------------------")

    def _log_response(self, response: httpx.Response) -> None:
        """
        Logs HTTP responses if debug mode is enabled.

        Args:
            response (httpx.Response): The HTTP response object.
        """

        logger.debug("-------------------------------------------")
        logger.debug("Response: %s", response.status_code)
        logger.debug("Headers: %s", response.headers)
        logger.debug("Content: %s", response.read())
        logger.debug("--------------------------------------------")

    def _get_error_message(self, response: httpx.Response) -> str:
        """
        Extracts the error message from an HTTP response.

        Args:
            response (httpx.Response): The HTTP response object.

        Returns:
            str: The error message extracted from the response.
        """

        try:
            obj = response.json()
            return f"{response.status_code}: {obj.get('detail', json.dumps(obj))}"
        except Exception:  # noqa: BLE001
            return f"{response.status_code}: {response.content!r}"

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, t.Any] | None = None,
        json_data: dict[str, t.Any] | None = None,
    ) -> httpx.Response:
        """
        Makes a raw HTTP request to the API.

        Args:
            method (str): The HTTP method (e.g., "GET", "POST").
            path (str): The API endpoint path.
            params (dict[str, Any] | None, optional): Query parameters for the request. Defaults to None.
            json_data (dict[str, Any] | None, optional): JSON payload for the request. Defaults to None.

        Returns:
            httpx.Response: The HTTP response object.
        """

        return self._client.request(method, path, json=json_data, params=params)

    def request(
        self,
        method: str,
        path: str,
        params: dict[str, t.Any] | None = None,
        json_data: dict[str, t.Any] | None = None,
    ) -> httpx.Response:
        """
        Makes an HTTP request to the API and raises exceptions for errors.

        Args:
            method (str): The HTTP method (e.g., "GET", "POST").
            path (str): The API endpoint path.
            params (dict[str, Any] | None, optional): Query parameters for the request. Defaults to None.
            json_data (dict[str, Any] | None, optional): JSON payload for the request. Defaults to None.

        Returns:
            httpx.Response: The HTTP response object.

        Raises:
            RuntimeError: If the response status code indicates an error.
        """

        response = self._request(method, path, params, json_data)

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(self._get_error_message(response)) from e

        return response

    # Auth

    def url_for_user_code(self, user_code: str) -> str:
        """Get the URL to verify the user code."""

        return f"{self._base_url.removesuffix('/api')}/account/device?code={user_code}"

    def get_device_codes(self) -> DeviceCodeResponse:
        """Start the authentication flow by requesting user and device codes."""

        response = self.request("POST", "/auth/device/code")
        return DeviceCodeResponse(**response.json())

    def poll_for_token(
        self,
        device_code: str,
        interval: int = DEFAULT_POLL_INTERVAL,
        max_poll_time: int = DEFAULT_MAX_POLL_TIME,
    ) -> AccessRefreshTokenResponse:
        """Poll for the access token with the given device code."""

        start_time = datetime.now(timezone.utc)
        while (datetime.now(timezone.utc) - start_time).total_seconds() < max_poll_time:
            response = self._request(
                "POST", "/auth/device/token", json_data={"device_code": device_code}
            )

            if response.status_code == 200:
                return AccessRefreshTokenResponse(**response.json())
            if response.status_code != 401:
                raise RuntimeError(self._get_error_message(response))

            time.sleep(interval)

        raise RuntimeError("Polling for token timed out")

    # User

    def get_user(self) -> UserResponse:
        """Get the user email and username."""

        response = self.request("GET", "/user")
        return UserResponse(**response.json())

    # Github

    def get_github_access_token(self, repos: list[str]) -> GithubTokenResponse:
        """Try to get a GitHub access token for the given repositories."""
        response = self.request("POST", "/github/token", json_data={"repos": repos})
        return GithubTokenResponse(**response.json())

    # Strikes

    def list_projects(self) -> list[Project]:
        """Retrieves a list of projects.

        Returns:
            list[Project]: A list of Project objects.
        """
        response = self.request("GET", "/strikes/projects")
        return [Project(**project) for project in response.json()]

    def get_project(self, project_identifier: str | UUID, workspace_id: UUID) -> Project:
        """Retrieves details of a specific project.

        Args:
            project_identifier (str | UUID): The project identifier. ID or key.

        Returns:
            Project: The Project object.
        """
        response = self.request(
            "GET",
            f"/strikes/projects/{project_identifier!s}",
            params={"workspace_id": workspace_id},
        )
        return Project(**response.json())

    def create_project(
        self,
        name: str,
        key: str,
        workspace_id: UUID | None = None,
        organization_id: UUID | None = None,
    ) -> Project:
        """Creates a new project.

        Args:
            name: The name of the project. If None, a default name will be used.
            workspace_id: The workspace ID to create the project in. If None, the default workspace will be used.
            organization_id: The organization ID to create the project in. If None, the default organization will be used.

        Returns:
            Project: The created Project object.
        """
        payload: dict[str, t.Any] = {}
        payload["name"] = name
        payload["key"] = key
        if workspace_id is not None:
            payload["workspace_id"] = str(workspace_id)
        if organization_id is not None:
            payload["org_id"] = str(organization_id)

        response = self.request("POST", "/strikes/projects", json_data=payload)
        return Project(**response.json())

    def list_runs(self, project: str) -> list[RunSummary]:
        """
        Lists all runs for a specific project.

        Args:
            project: The project identifier.

        Returns:
            A list of RunSummary objects representing the runs in the project.
        """
        response = self.request("GET", f"/strikes/projects/{project!s}/runs")
        return [RunSummary(**run) for run in response.json()]

    def _get_run(self, run: str | ULID) -> RawRun:
        response = self.request("GET", f"/strikes/projects/runs/{run!s}")
        return RawRun(**response.json())

    def get_run(self, run: str | ULID) -> Run:
        """
        Retrieves details of a specific run.

        Args:
            run: The run identifier.

        Returns:
            The Run object containing details of the run.
        """
        return process_run(self._get_run(run))

    TraceFormat = t.Literal["tree", "flat"]

    @t.overload
    def get_run_tasks(self, run: str | ULID, *, format: t.Literal["tree"]) -> list[TaskTree]: ...

    @t.overload
    def get_run_tasks(
        self, run: str | ULID, *, format: t.Literal["flat"] = "flat"
    ) -> list[Task]: ...

    def get_run_tasks(
        self, run: str | ULID, *, format: TraceFormat = "flat"
    ) -> list[Task] | list[TaskTree]:
        """
        Gets all tasks for a specific run.

        Args:
            run: The run identifier.
            format: The format of the tasks to return. Can be "flat" or "tree".

        Returns:
            A list of Task objects in flat format or a list of TaskTree objects in tree format.
        """
        raw_run = self._get_run(run)
        response = self.request("GET", f"/strikes/projects/runs/{run!s}/tasks/full")
        raw_tasks = [RawTask(**task) for task in response.json()]
        tasks = [process_task(task, raw_run) for task in raw_tasks]
        tasks = sorted(tasks, key=lambda x: x.timestamp)
        return tasks if format == "flat" else convert_flat_tasks_to_tree(tasks)

    @t.overload
    def get_run_trace(self, run: str | ULID, *, format: t.Literal["tree"]) -> list[TraceTree]: ...

    @t.overload
    def get_run_trace(
        self, run: str | ULID, *, format: t.Literal["flat"] = "flat"
    ) -> list[Task | TraceSpan]: ...

    def get_run_trace(
        self, run: str | ULID, *, format: TraceFormat = "flat"
    ) -> list[Task | TraceSpan] | list[TraceTree]:
        """
        Retrieves the run trace (spans+tasks) of a specific run.

        Args:
            run: The run identifier.
            format: The format of the trace to return. Can be "flat" or "tree".

        Returns:
            A list of Task or TraceSpan objects in flat format or a list of TraceTree objects in tree format.
        """
        raw_run = self._get_run(run)
        response = self.request("GET", f"/strikes/projects/runs/{run!s}/spans/full")
        trace: list[Task | TraceSpan] = []
        for item in response.json():
            if "parent_task_span_id" in item:
                trace.append(process_task(RawTask(**item), raw_run))
            else:
                trace.append(TraceSpan(**item))

        trace = sorted(trace, key=lambda x: x.timestamp)
        return trace if format == "flat" else convert_flat_trace_to_tree(trace)

    # Data Exports

    def export_metrics(
        self,
        project: str,
        *,
        filter: str | None = None,
        # format: ExportFormat = "parquet",
        status: StatusFilter = "completed",
        metrics: list[str] | None = None,
        aggregations: list[MetricAggregationType] | None = None,
    ) -> "pd.DataFrame":
        """
        Exports metric data for a specific project.

        Args:
            project: The project identifier.
            filter: A filter to apply to the exported data. Defaults to None.
            status: The status of metrics to include. Defaults to "completed".
            metrics: A list of metric names to include. Defaults to None.
            aggregations: A list of aggregation types to apply. Defaults to None.

        Returns:
            A DataFrame containing the exported metric data.
        """
        import pandas as pd

        response = self.request(
            "GET",
            f"/strikes/projects/{project!s}/export/metrics",
            params={
                "format": "parquet",
                "status": status,
                "filter": filter,
                **({"metrics": metrics} if metrics else {}),
                **({"aggregations": aggregations} if aggregations else {}),
            },
        )
        return pd.read_parquet(io.BytesIO(response.content))

    def export_runs(
        self,
        project: str,
        *,
        filter: str | None = None,
        status: StatusFilter = "completed",
        aggregations: list[MetricAggregationType] | None = None,
        format: ExportFormat = "parquet",
        base_dir: str | None = None,
    ) -> str:
        """
        Export runs using pagination - always writes to disk.

        Args:
            project: The project identifier
            filter: A filter to apply to the exported data
            status: The status of runs to include
            aggregations: A list of aggregation types to apply
            format: Output format - "parquet", "csv", "json", "jsonl"
            base_dir: Base directory for export (defaults to "./strikes-data")

        Returns:
            str: Path to the export directory
        """
        import pandas as pd

        logger.info(f"Starting paginated export for project '{project}', format='{format}'")

        page = 1
        first_response = self.request(
            "GET",
            f"/strikes/projects/{project!s}/export/paginated",
            params={
                "page": page,
                "status": status,
                **({"filter": filter} if filter else {}),
                **({"aggregations": aggregations} if aggregations else {}),
            },
        )

        if not first_response.content:
            logger.info("No data found")

        first_chunk = pd.read_parquet(io.BytesIO(first_response.content))

        total_runs = int(first_response.headers.get("x-total", "0"))
        has_more = first_response.headers.get("x-has-more", "false") == "true"

        logger.info(f"Total runs: {total_runs}, Has more: {has_more}")

        logger.info(f"Writing {total_runs} runs to disk")
        return self._export_to_disk(
            project,
            first_chunk,
            dict(first_response.headers),
            filter,
            status,
            aggregations,
            format,
            str(base_dir) if base_dir else None,
        )

    def _export_to_disk(
        self,
        project: str,
        first_chunk: "pd.DataFrame",
        first_headers: dict[str, str],
        filter: str | None,
        status: StatusFilter,
        aggregations: list[MetricAggregationType] | None,
        format: str,
        base_dir: str | None,
    ) -> str:
        """Handle dataset export to disk - one file per chunk."""
        import pandas as pd

        if base_dir:
            export_base = Path(base_dir) / "strikes-data" / "export-runs"
        else:
            export_base = Path("./strikes-data") / "export-runs"

        project_dir = export_base / project

        logger.info(f"Using project name: '{project}'")
        logger.info(f"Project directory will be: {project_dir}")

        # Clean up old project data
        if project_dir.exists():
            logger.info(f"Removing old export data: {project_dir}")
            shutil.rmtree(project_dir)

        project_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {project_dir}")

        page = 1
        total_exported_runs = 0

        # Write first chunk
        filename = self._write_chunk_file(first_chunk, project_dir, page, format)
        chunk_run_count = len(first_chunk["run_id"].unique())
        total_exported_runs += chunk_run_count
        logger.info(f"Page {page}: Wrote {filename} ({chunk_run_count} runs)")

        has_more = first_headers.get("x-has-more", "false") == "true"
        total_runs = int(first_headers.get("x-total", "0"))

        logger.info(f"Total runs to export: {total_runs}")

        # Loop through remaining pages - SDK just increments page until has_more = false
        while has_more:
            page += 1
            logger.info(f"Fetching page {page}")

            try:
                response = self.request(
                    "GET",
                    f"/strikes/projects/{project!s}/export/paginated",
                    params={
                        "page": page,
                        "status": status,
                        **({"filter": filter} if filter else {}),
                        **({"aggregations": aggregations} if aggregations else {}),
                    },
                )

                if not response.content:
                    logger.info("No more data - empty response")
                    break

                # Parse response
                chunk_df = pd.read_parquet(io.BytesIO(response.content))

                if chunk_df.empty:
                    logger.info("Empty chunk received - breaking")
                    break

                # Write chunk
                filename = self._write_chunk_file(chunk_df, project_dir, page, format)
                chunk_run_count = len(chunk_df["run_id"].unique())
                total_exported_runs += chunk_run_count
                logger.info(f"Page {page}: Wrote {filename} ({chunk_run_count} runs)")

                # Check if API has more pages
                has_more = response.headers.get("x-has-more", "false") == "true"

            except Exception as e:  # noqa: BLE001
                logger.error(f"Error fetching page {page}: {e}")
                break

        logger.info(f"Export complete to {project_dir}")
        logger.info(f"Total pages: {page}, Total runs: {total_exported_runs}")

        return str(project_dir)

    def _write_chunk_file(
        self, df: "pd.DataFrame", project_dir: Path, page: int, format: str
    ) -> str:
        """Write chunk to a single file with intelligent naming."""

        if df.empty:
            return ""

        total_runs = len(df["run_id"].unique())
        total_rows = len(df)

        logger.info(f"Writing chunk: {total_rows} rows, {total_runs} unique runs, page {page}")

        if total_runs == 1:
            # Single run - use the run ID
            run_id = df["run_id"].iloc[0]
            base_name = f"run_{run_id}"
        elif total_runs <= 10:
            # Few runs - include count
            base_name = f"runs_{total_runs}_page_{page}"
        elif total_runs <= 100:
            # Medium batch - include count
            base_name = f"runs_{total_runs}_batch_{page}"
        else:
            base_name = f"runs_chunk_{page}_{total_runs}_runs"

        # Write based on format
        if format == "csv":
            filename = f"{base_name}.csv"
            filepath = project_dir / filename
            df.to_csv(filepath, index=False)
            logger.info(f"Wrote CSV: {filepath}")
        elif format == "parquet":
            filename = f"{base_name}.parquet"
            filepath = project_dir / filename
            df.to_parquet(filepath, index=False)
            logger.info(f"Wrote Parquet: {filepath}")
        elif format == "json":
            filename = f"{base_name}.json"
            filepath = project_dir / filename
            df.to_json(filepath, orient="records", indent=2)
            logger.info(f"Wrote JSON: {filepath}")
        elif format == "jsonl":
            filename = f"{base_name}.jsonl"
            filepath = project_dir / filename
            df.to_json(filepath, orient="records", lines=True)
            logger.info(f"Wrote JSONL: {filepath}")
        else:
            # Default to parquet
            filename = f"{base_name}.parquet"
            filepath = project_dir / filename
            df.to_parquet(filepath, index=False)
            logger.info(f"Wrote default Parquet: {filepath}")

        return filename

    def export_parameters(
        self,
        project: str,
        *,
        filter: str | None = None,
        # format: ExportFormat = "parquet",
        status: StatusFilter = "completed",
        parameters: list[str] | None = None,
        metrics: list[str] | None = None,
        aggregations: list[MetricAggregationType] | None = None,
    ) -> "pd.DataFrame":
        """
        Exports parameter data for a specific project.

        Args:
            project: The project identifier.
            filter: A filter to apply to the exported data. Defaults to None.
            status: The status of parameters to include. Defaults to "completed".
            parameters: A list of parameter names to include. Defaults to None.
            metrics: A list of metric names to include. Defaults to None.
            aggregations: A list of aggregation types to apply. Defaults to None.

        Returns:
            A DataFrame containing the exported parameter data.
        """
        import pandas as pd

        response = self.request(
            "GET",
            f"/strikes/projects/{project!s}/export/parameters",
            params={
                "format": "parquet",
                "status": status,
                "filter": filter,
                **({"parameters": parameters} if parameters else {}),
                **({"metrics": metrics} if metrics else {}),
                **({"aggregations": aggregations} if aggregations else {}),
            },
        )
        return pd.read_parquet(io.BytesIO(response.content))

    def export_timeseries(
        self,
        project: str,
        *,
        filter: str | None = None,
        # format: ExportFormat = "parquet",
        status: StatusFilter = "completed",
        metrics: list[str] | None = None,
        time_axis: TimeAxisType = "relative",
        aggregations: list[TimeAggregationType] | None = None,
    ) -> "pd.DataFrame":
        """
        Exports timeseries data for a specific project.

        Args:
            project: The project identifier.
            filter: A filter to apply to the exported data. Defaults to None.
            status: The status of timeseries to include. Defaults to "completed".
            metrics: A list of metric names to include. Defaults to None.
            time_axis: The type of time axis to use. Defaults to "relative".
            aggregations: A list of aggregation types to apply. Defaults to None.

        Returns:
            A DataFrame containing the exported timeseries data.
        """
        import pandas as pd

        response = self.request(
            "GET",
            f"/strikes/projects/{project!s}/export/timeseries",
            params={
                "format": "parquet",
                "status": status,
                "filter": filter,
                "time_axis": time_axis,
                **({"metrics": metrics} if metrics else {}),
                **({"aggregation": aggregations} if aggregations else {}),
            },
        )
        return pd.read_parquet(io.BytesIO(response.content))

    # User data access

    def get_user_data_credentials(
        self, organization_id: UUID, workspace_id: UUID
    ) -> UserDataCredentials:
        """
        Retrieves user data credentials for secondary storage access.

        Returns:
            The user data credentials object.
        """
        params = {"org_id": str(organization_id), "workspace_id": str(workspace_id)}
        response = self._request("GET", "/user-data/credentials", params=params)
        return UserDataCredentials(**response.json())

    # Container registry access

    def get_platform_registry_credentials(self) -> ContainerRegistryCredentials:
        """
        Retrieves container registry credentials for Docker image access.

        Returns:
            The container registry credentials object.
        """
        response = self.request("POST", "/platform/registry-token")
        return ContainerRegistryCredentials(**response.json())

    def get_platform_releases(self, tag: str, services: list[str]) -> RegistryImageDetails:
        """
        Resolves the platform releases for the current project.

        Returns:
            The resolved platform releases as a ResolveReleasesResponse object.
        """
        from dreadnode.version import VERSION

        payload = {"tag": tag, "services": services, "cli_version": VERSION}
        response = self.request("POST", "/platform/get-releases", json_data=payload)
        return RegistryImageDetails(**response.json())

    def get_platform_templates(self, tag: str) -> bytes:
        """
        Retrieves the available platform templates.
        """
        params = {"tag": tag}
        response = self.request("GET", "/platform/templates/all", params=params)
        zip_content: bytes = response.content
        return zip_content

    # RBAC
    def list_organizations(self) -> list[Organization]:
        """
        Retrieves a list of organizations the user belongs to.

        Returns:
            A list of organization names.
        """
        response = self.request("GET", "/organizations")
        return [Organization(**org) for org in response.json()]

    def get_organization(self, org_id_or_key: UUID | str) -> Organization:
        """
        Retrieves details of a specific organization.

        Args:
            org_id_or_key (str | UUID): The organization identifier.

        Returns:
            Organization: The Organization object.
        """
        response = self.request("GET", f"/organizations/{org_id_or_key!s}")
        return Organization(**response.json())

    def list_workspaces(self, filters: WorkspaceFilter | None = None) -> list[Workspace]:
        """
        Retrieves a list of workspaces the user has access to.

        Returns:
            A list of workspace names.
        """
        response = self.request(
            "GET", "/workspaces", params=filters.model_dump() if filters else None
        )
        paginated_workspaces = PaginatedWorkspaces(**response.json())
        # handle the pagination
        all_workspaces: list[Workspace] = paginated_workspaces.workspaces.copy()
        while paginated_workspaces.has_next:
            response = self.request(
                "GET",
                "/workspaces",
                params={
                    "page": paginated_workspaces.page + 1,
                    "limit": paginated_workspaces.limit,
                    **(filters.model_dump() if filters else {}),
                },
            )
            next_page = PaginatedWorkspaces(**response.json())
            all_workspaces.extend(next_page.workspaces)
            paginated_workspaces.page = next_page.page
            paginated_workspaces.has_next = next_page.has_next

        return all_workspaces

    def get_workspace(
        self, workspace_id_or_key: UUID | str, org_id: UUID | None = None
    ) -> Workspace:
        """
        Retrieves details of a specific workspace.

        Args:
            workspace_id_or_key (str | UUID): The workspace identifier.

        Returns:
            Workspace: The Workspace object.
        """
        params: dict[str, str] = {}
        if org_id:
            params = {"org_id": str(org_id)}
        response = self.request("GET", f"/workspaces/{workspace_id_or_key!s}", params=params)
        return Workspace(**response.json())

    def create_workspace(
        self,
        name: str,
        key: str,
        organization_id: UUID,
        description: str | None = None,
    ) -> Workspace:
        """
        Creates a new workspace.

        Args:
            name (str): The name of the workspace.
            organization_id (str | UUID): The organization ID to create the workspace in.

        Returns:
            Workspace: The created Workspace object.
        """

        payload = {
            "name": name,
            "key": key,
            "description": description,
            "org_id": str(organization_id),
        }

        response = self.request("POST", "/workspaces", json_data=payload)
        return Workspace(**response.json())

    def delete_workspace(self, workspace_id: str | UUID) -> None:
        """
        Deletes a specific workspace.

        Args:
            workspace_id (str | UUID): The workspace key.
        """

        self.request("DELETE", f"/workspaces/{workspace_id!s}")
