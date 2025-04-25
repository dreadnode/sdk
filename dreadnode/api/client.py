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
    """Client for the Dreadnode API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        debug: bool = False,
    ):
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
        """Log every request to the console if debug is enabled."""

        logger.debug("-------------------------------------------")
        logger.debug("%s %s", request.method, request.url)
        logger.debug("Headers: %s", request.headers)
        logger.debug("Content: %s", request.content)
        logger.debug("-------------------------------------------")

    def _log_response(self, response: httpx.Response) -> None:
        """Log every response to the console if debug is enabled."""

        logger.debug("-------------------------------------------")
        logger.debug("Response: %s", response.status_code)
        logger.debug("Headers: %s", response.headers)
        logger.debug("Content: %s", response.read())
        logger.debug("--------------------------------------------")

    def _get_error_message(self, response: httpx.Response) -> str:
        """Get the error message from the response."""

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
        """Make a raw request to the API."""

        return self._client.request(method, path, json=json_data, params=params)

    def request(
        self,
        method: str,
        path: str,
        params: dict[str, t.Any] | None = None,
        json_data: dict[str, t.Any] | None = None,
    ) -> httpx.Response:
        """Make a request to the API. Raise an exception for non-200 status codes."""

        response = self._request(method, path, params, json_data)
        if response.status_code == 401:  # noqa: PLR2004
            raise RuntimeError("Authentication failed, please check your API token.")

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(self._get_error_message(response)) from e

        return response

    def list_projects(self) -> list[Project]:
        response = self.request("GET", "/strikes/projects")
        return [Project(**project) for project in response.json()]

    def get_project(self, project: str) -> Project:
        response = self.request("GET", f"/strikes/projects/{project!s}")
        return Project(**response.json())

    def list_runs(self, project: str) -> list[Run]:
        response = self.request("GET", f"/strikes/projects/{project!s}/runs")
        return [Run(**run) for run in response.json()]

    def get_run(self, run: str | ULID) -> Run:
        response = self.request("GET", f"/strikes/projects/runs/{run!s}")
        return Run(**response.json())

    def get_run_tasks(self, run: str | ULID) -> list[Task]:
        response = self.request("GET", f"/strikes/projects/runs/{run!s}/tasks")
        return [Task(**task) for task in response.json()]

    def get_run_trace(self, run: str | ULID) -> list[Task | TraceSpan]:
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
        # format: ExportFormat = "parquet",
        status: StatusFilter = "completed",
        aggregations: list[MetricAggregationType] | None = None,
    ) -> pd.DataFrame:
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
        response = self.request("GET", "/user-data/credentials")
        return UserDataCredentials(**response.json())
