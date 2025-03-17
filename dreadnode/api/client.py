import json
import typing as t

import httpx
from pydantic import BaseModel
from rich import print
from ulid import ULID

from dreadnode.version import VERSION

from .models import Project, Run, Task, TraceSpan

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

        print("-------------------------------------------")
        print(f"[bold]{request.method}[/] {request.url}")
        print("Headers:", request.headers)
        print("Content:", request.content)
        print("-------------------------------------------")

    def _log_response(self, response: httpx.Response) -> None:
        """Log every response to the console if debug is enabled."""

        print("-------------------------------------------")
        print(f"Response: {response.status_code}")
        print("Headers:", response.headers)
        print("Content:", response.read())
        print("--------------------------------------------")

    def _get_error_message(self, response: httpx.Response) -> str:
        """Get the error message from the response."""

        try:
            obj = response.json()
            return f'{response.status_code}: {obj.get("detail", json.dumps(obj))}'
        except Exception:  # noqa: BLE001
            return str(response.content)

    def _request(
        self,
        method: str,
        path: str,
        query_params: dict[str, str] | None = None,
        json_data: dict[str, t.Any] | None = None,
    ) -> httpx.Response:
        """Make a raw request to the API."""

        return self._client.request(method, path, json=json_data, params=query_params)

    def request(
        self,
        method: str,
        path: str,
        query_params: dict[str, str] | None = None,
        json_data: dict[str, t.Any] | None = None,
    ) -> httpx.Response:
        """Make a request to the API. Raise an exception for non-200 status codes."""

        response = self._request(method, path, query_params, json_data)
        if response.status_code == 401:  # noqa: PLR2004
            raise RuntimeError("Authentication failed, please check your API token.")

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(self._get_error_message(response)) from e

        return response

    def list_projects(self) -> list[Project]:
        response = self._client.request("GET", "/strikes/projects")
        return [Project(**project) for project in response.json()]

    def get_project(self, project: str) -> Project:
        response = self._client.request("GET", f"/strikes/projects/{project!s}")
        return Project(**response.json())

    def list_runs(self, project: str) -> list[Run]:
        response = self._client.request("GET", f"/strikes/projects/{project!s}/runs")
        return [Run(**run) for run in response.json()]

    def get_run(self, run: str | ULID) -> Run:
        response = self._client.request("GET", f"/strikes/projects/runs/{run!s}")
        return Run(**response.json())

    def get_run_tasks(self, run: str | ULID) -> list[Task]:
        response = self._client.request("GET", f"/strikes/projects/runs/{run!s}/tasks")
        return [Task(**task) for task in response.json()]

    def get_run_trace(self, run: str | ULID) -> list[Task | TraceSpan]:
        response = self._client.request("GET", f"/strikes/projects/runs/{run!s}/spans")
        spans: list[Task | TraceSpan] = []
        for item in response.json():
            if "parent_task_span_id" in item:
                spans.append(Task(**item))
            else:
                spans.append(TraceSpan(**item))
        return spans
