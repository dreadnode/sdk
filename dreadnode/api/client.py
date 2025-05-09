"""
This module provides the `ApiClient` class, which serves as a client for interacting
with the Dreadnode API. It includes methods for managing projects, runs, tasks, and
exporting data in various formats.

Dependencies:
    - httpx
    - pandas
    - pydantic
    - ulid
    - dreadnode.util.logger
    - dreadnode.version.VERSION
"""

import io
import json
import typing as t

import httpx
import pandas as pd
from pydantic import BaseModel
from ulid import ULID

from dreadnode.util import logger
from dreadnode.version import VERSION

from .models import (
    MetricAggregationType,
    Project,
    Run,
    StatusFilter,
    Task,
    TimeAggregationType,
    TimeAxisType,
    TraceSpan,
    UserDataCredentials,
)

ModelT = t.TypeVar("ModelT", bound=BaseModel)


class ApiClient:
    """Client for the Dreadnode API.

    This class provides methods to interact with the Dreadnode API, including
    managing projects, runs, tasks, and exporting data.

    Attributes:
        _base_url (str): The base URL of the API.
        _client (httpx.Client): The HTTP client used for making requests.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        debug: bool = False,
    ):
        """Initializes the ApiClient.

        Args:
            base_url (str): The base URL of the API.
            api_key (str): The API key for authentication.
            debug (bool, optional): Whether to enable debug logging. Defaults to False.
        """
        self._base_url = base_url.rstrip("/")
        if not self._base_url.endswith("/api"):
            self._base_url += "/api"

        self._client = httpx.Client(
            headers={
                "User-Agent": f"dreadnode-sdk/{VERSION}",
                "Accept": "application/json",
                "X-API-Key": api_key,
            },
            base_url=self._base_url,
            timeout=30,
        )

        if debug:
            self._client.event_hooks["request"].append(self._log_request)
            self._client.event_hooks["response"].append(self._log_response)

    def _log_request(self, request: httpx.Request) -> None:
        """Logs HTTP requests if debug is enabled.

        Args:
            request (httpx.Request): The HTTP request object.
        """
        logger.debug("-------------------------------------------")
        logger.debug("%s %s", request.method, request.url)
        logger.debug("Headers: %s", request.headers)
        logger.debug("Content: %s", request.content)
        logger.debug("-------------------------------------------")

    def _log_response(self, response: httpx.Response) -> None:
        """Logs HTTP responses if debug is enabled.

        Args:
            response (httpx.Response): The HTTP response object.
        """
        logger.debug("-------------------------------------------")
        logger.debug("Response: %s", response.status_code)
        logger.debug("Headers: %s", response.headers)
        logger.debug("Content: %s", response.read())
        logger.debug("--------------------------------------------")

    def _get_error_message(self, response: httpx.Response) -> str:
        """Extracts the error message from an HTTP response.

        Args:
            response (httpx.Response): The HTTP response object.

        Returns:
            str: The error message.
        """
        try:
            obj = response.json()
            return f"{response.status_code}: {obj.get('detail', json.dumps(obj))}"
        except Exception:
            return str(response.content)

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, t.Any] | None = None,
        json_data: dict[str, t.Any] | None = None,
    ) -> httpx.Response:
        """Makes a raw HTTP request to the API.

        Args:
            method (str): The HTTP method (e.g., "GET", "POST").
            path (str): The API endpoint path.
            params (dict[str, t.Any], optional): Query parameters. Defaults to None.
            json_data (dict[str, t.Any], optional): JSON payload. Defaults to None.

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
        """Makes an HTTP request to the API and raises exceptions for errors.

        Args:
            method (str): The HTTP method (e.g., "GET", "POST").
            path (str): The API endpoint path.
            params (dict[str, t.Any], optional): Query parameters. Defaults to None.
            json_data (dict[str, t.Any], optional): JSON payload. Defaults to None.

        Returns:
            httpx.Response: The HTTP response object.

        Raises:
            RuntimeError: If the response status code indicates an error.
        """
        response = self._request(method, path, params, json_data)
        if response.status_code == 401:
            raise RuntimeError("Authentication failed, please check your API token.")

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(self._get_error_message(response)) from e

        return response

    def list_projects(self) -> list[Project]:
        """Lists all projects.

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

    def list_runs(self, project: str) -> list[Run]:
        """Lists all runs for a specific project.

        Args:
            project (str): The project identifier.

        Returns:
            list[Run]: A list of Run objects.
        """
        response = self.request("GET", f"/strikes/projects/{project!s}/runs")
        return [Run(**run) for run in response.json()]

    def get_run(self, run: str | ULID) -> Run:
        """Retrieves details of a specific run.

        Args:
            run (str | ULID): The run identifier.

        Returns:
            Run: The Run object.
        """
        response = self.request("GET", f"/strikes/projects/runs/{run!s}")
        return Run(**response.json())

    def get_run_tasks(self, run: str | ULID) -> list[Task]:
        """Lists all tasks for a specific run.

        Args:
            run (str | ULID): The run identifier.

        Returns:
            list[Task]: A list of Task objects.
        """
        response = self.request("GET", f"/strikes/projects/runs/{run!s}/tasks")
        return [Task(**task) for task in response.json()]

    def get_run_trace(self, run: str | ULID) -> list[Task | TraceSpan]:
        """Retrieves the trace spans for a specific run.

        Args:
            run (str | ULID): The run identifier.

        Returns:
            list[Task | TraceSpan]: A list of Task or TraceSpan objects.
        """
        response = self.request("GET", f"/strikes/projects/runs/{run!s}/spans")
        spans: list[Task | TraceSpan] = []
        for item in response.json():
            if "parent_task_span_id" in item:
                spans.append(Task(**item))
            else:
                spans.append(TraceSpan(**item))
        return spans

    # Data exports

    def export_runs(
        self,
        project: str,
        *,
        filter: str | None = None,
        status: StatusFilter = "completed",
        aggregations: list[MetricAggregationType] | None = None,
    ) -> pd.DataFrame:
        """Exports run data for a specific project.

        Args:
            project (str): The project identifier.
            filter (str, optional): A filter string. Defaults to None.
            status (StatusFilter, optional): The status filter. Defaults to "completed".
            aggregations (list[MetricAggregationType], optional): Aggregations to apply. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the exported data.
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
        status: StatusFilter = "completed",
        metrics: list[str] | None = None,
        aggregations: list[MetricAggregationType] | None = None,
    ) -> pd.DataFrame:
        """Exports metric data for a specific project.

        Args:
            project (str): The project identifier.
            filter (str, optional): A filter string. Defaults to None.
            status (StatusFilter, optional): The status filter. Defaults to "completed".
            metrics (list[str], optional): A list of metrics to include. Defaults to None.
            aggregations (list[MetricAggregationType], optional): Aggregations to apply. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the exported data.
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
        status: StatusFilter = "completed",
        parameters: list[str] | None = None,
        metrics: list[str] | None = None,
        aggregations: list[MetricAggregationType] | None = None,
    ) -> pd.DataFrame:
        """Exports parameter data for a specific project.

        Args:
            project (str): The project identifier.
            filter (str, optional): A filter string. Defaults to None.
            status (StatusFilter, optional): The status filter. Defaults to "completed".
            parameters (list[str], optional): A list of parameters to include. Defaults to None.
            metrics (list[str], optional): A list of metrics to include. Defaults to None.
            aggregations (list[MetricAggregationType], optional): Aggregations to apply. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the exported data.
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
        status: StatusFilter = "completed",
        metrics: list[str] | None = None,
        time_axis: TimeAxisType = "relative",
        aggregations: list[TimeAggregationType] | None = None,
    ) -> pd.DataFrame:
        """Exports timeseries data for a specific project.

        Args:
            project (str): The project identifier.
            filter (str, optional): A filter string. Defaults to None.
            status (StatusFilter, optional): The status filter. Defaults to "completed".
            metrics (list[str], optional): A list of metrics to include. Defaults to None.
            time_axis (TimeAxisType, optional): The time axis type. Defaults to "relative".
            aggregations (list[TimeAggregationType], optional): Aggregations to apply. Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the exported data.
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
        """Retrieves user data credentials.

        Returns:
            UserDataCredentials: The user data credentials.
        """
        response = self.request("GET", "/user-data/credentials")
        return UserDataCredentials(**response.json())
