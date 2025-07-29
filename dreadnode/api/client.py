import io
import json
import time
import typing as t
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
import pandas as pd
from pydantic import BaseModel
from ulid import ULID

from dreadnode.api.models import (
    AccessRefreshTokenResponse,
    DeviceCodeResponse,
    GithubTokenResponse,
    MetricAggregationType,
    Project,
    RawRun,
    RawTask,
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
from dreadnode.util import logger
from dreadnode.version import VERSION

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
    ):
        """
        Initializes the API client.

        Args:
            base_url (str): The base URL of the Dreadnode API.
            api_key (str): The API key for authentication.
            debug (bool, optional): Whether to enable debug logging. Defaults to False.
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
            headers["X-Api-Key"] = api_key

        self._client = httpx.Client(
            headers=headers,
            cookies=_cookies,
            base_url=self._base_url,
            timeout=30,
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
            return str(response.content)

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

            if response.status_code == 200:  # noqa: PLR2004
                return AccessRefreshTokenResponse(**response.json())
            if response.status_code != 401:  # noqa: PLR2004
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

    def get_project(self, project: str) -> Project:
        """Retrieves details of a specific project.

        Args:
            project (str): The project identifier.

        Returns:
            Project: The Project object.
        """
        response = self.request("GET", f"/strikes/projects/{project!s}")
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

    # Data exports

    def export_runs(
        self,
        project: str,
        *,
        filter: str | None = None,
        # format: ExportFormat = "parquet",
        status: StatusFilter = "completed",
        aggregations: list[MetricAggregationType] | None = None,
    ) -> pd.DataFrame:
        """
        Exports run data for a specific project.

        Args:
            project: The project identifier.
            filter: A filter to apply to the exported data. Defaults to None.
            status: The status of runs to include. Defaults to "completed".
            aggregations: A list of aggregation types to apply. Defaults to None.

        Returns:
            A DataFrame containing the exported run data.
        """
        response = self.request(
            "GET",
            f"/strikes/projects/{project!s}/export",
            params={
                "format": "parquet",
                "status": status,
                **({"filter": filter} if filter else {}),
                **({"aggregations": aggregations} if aggregations else {}),
            },
        )
        return pd.read_parquet(io.BytesIO(response.content))

    def export_metrics(
        self,
        project: str,
        *,
        filter: str | None = None,
        # format: ExportFormat = "parquet",
        status: StatusFilter = "completed",
        metrics: list[str] | None = None,
        aggregations: list[MetricAggregationType] | None = None,
    ) -> pd.DataFrame:
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
    ) -> pd.DataFrame:
        """
        Exports parameter data for a specific project.

        Args:
            project: The project identifier.
            filter: A filter to apply to the exported data. Defaults to None.
            status : The status of parameters to include. Defaults to "completed".
            parameters: A list of parameter names to include. Defaults to None.
            metrics: A list of metric names to include. Defaults to None.
            aggregations: A list of aggregation types to apply. Defaults to None.

        Returns:
            A DataFrame containing the exported parameter data.
        """
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
    ) -> pd.DataFrame:
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

    def get_user_data_credentials(self) -> UserDataCredentials:
        """
        Retrieves user data credentials for secondary storage access.

        Returns:
            The user data credentials object.
        """
        response = self._request("GET", "/user-data/credentials")
        return UserDataCredentials(**response.json())
