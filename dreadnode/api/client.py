import json
import typing as t

import httpx
from pydantic import BaseModel
from rich import print

from ..version import VERSION
from .strikes import StrikesClient

ModelT = t.TypeVar("ModelT", bound=BaseModel)


class ApiClient:
    """Client for the Dreadnode API."""

    strikes: StrikesClient

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

        self.strikes = StrikesClient(self)

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
        except Exception:
            return str(response.content)

    def _request(
        self,
        method: str,
        path: str,
        query_params: dict[str, t.Any] | None = None,
        json_data: dict[str, t.Any] | None = None,
    ) -> httpx.Response:
        """Make a raw request to the API."""

        return self._client.request(method, path, json=json_data, params=query_params)

    def request(
        self,
        method: str,
        path: str,
        query_params: dict[str, t.Any] | None = None,
        json_data: dict[str, t.Any] | None = None,
    ) -> httpx.Response:
        """Make a request to the API. Raise an exception for non-200 status codes."""

        response = self._request(method, path, query_params, json_data)
        if response.status_code == 401:
            raise Exception("Authentication failed, please check your API token.")

        try:
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            raise Exception(self._get_error_message(response)) from e
