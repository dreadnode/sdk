import json
import typing as t

import httpx
from pydantic import BaseModel
from rich import print

from ..version import VERSION  # Assuming this import path is correct
from .strikes import StrikesClient  # Assuming this import path is correct

ModelT = t.TypeVar("ModelT", bound=BaseModel)


class ApiClient:
    """Client for interacting with the Dreadnode API.

    Provides methods for making HTTP requests to the Dreadnode backend,
    handling authentication, base URL configuration, and optional debugging logs.
    It also exposes specialized clients for different API resource groups,
    like 'strikes'.

    Attributes:
        strikes: An instance of StrikesClient for interacting with the strikes API.
    """

    strikes: StrikesClient

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        debug: bool = False,
    ):
        """Initializes the ApiClient.

        Sets up the base URL, API key for authentication, and configures
        an httpx client for making requests. Optionally enables debug logging.

        Args:
            base_url: The base URL for the Dreadnode API server.
                      It will be adjusted to ensure it ends with '/api'.
            api_key: The secret API key used for authenticating requests via
                     the 'X-API-Key' header.
            debug: If True, request and response details will be printed to
                   the console using 'rich'. Defaults to False.
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

        # Initialize sub-clients after the main client is ready
        self.strikes = StrikesClient(self)

    def _log_request(self, request: httpx.Request) -> None:
        """Logs request details to the console when debug is enabled.

        Uses 'rich' for formatted printing. Includes method, URL, headers,
        and request body content.

        Args:
            request: The httpx.Request object being sent.
        """
        print("------------------ Request ------------------")
        print(f"[bold]{request.method}[/] {request.url}")
        print("Headers:", dict(request.headers)) # Convert headers to dict for cleaner print
        # Attempt to decode content if possible, otherwise show bytes
        try:
            content = request.content.decode('utf-8')
            # Try parsing as JSON for pretty printing
            try:
                 content = json.dumps(json.loads(content), indent=2)
            except json.JSONDecodeError:
                 pass # Keep as string if not valid JSON
        except UnicodeDecodeError:
             content = request.content # Show bytes if not decodable
        print("Content:", content)
        print("-------------------------------------------")

    def _log_response(self, response: httpx.Response) -> None:
        """Logs response details to the console when debug is enabled.

        Uses 'rich' for formatted printing. Includes status code, headers,
        and response body content. Reads the response content to make it
        available for logging.

        Args:
            response: The httpx.Response object received.
        """
        # Ensure response content is read before logging it
        # This is crucial as httpx response bodies are stream-like
        response.read()

        print("------------------ Response -----------------")
        print(f"Status Code: {response.status_code}")
        print("Headers:", dict(response.headers)) # Convert headers to dict
        # Attempt to decode content if possible, otherwise show bytes
        try:
            content = response.text
             # Try parsing as JSON for pretty printing
            try:
                 content = json.dumps(json.loads(content), indent=2)
            except json.JSONDecodeError:
                 pass # Keep as string if not valid JSON
        except Exception: # Broad exception for any decoding/reading issue
             content = response.content # Fallback to bytes

        print("Content:", content)
        print("--------------------------------------------")

    def _get_error_message(self, response: httpx.Response) -> str:
        """Extracts a user-friendly error message from an HTTP response.

        Attempts to parse the response body as JSON and extract the 'detail'
        field. Falls back to the raw response content if JSON parsing fails.

        Args:
            response: The httpx.Response object representing the error.

        Returns:
            A string containing the status code and a descriptive error detail.
        """
        try:
            obj = response.json()
            detail = obj.get("detail", json.dumps(obj)) # Prefer 'detail', else dump whole object
            return f'{response.status_code}: {detail}'
        except json.JSONDecodeError:
            # If response is not JSON, return status code and raw content
            return f"{response.status_code}: {response.text}"
        except Exception as e:
             # Catch other potential errors during processing
             return f"{response.status_code}: Error processing response - {str(e)}"

    def _request(
        self,
        method: str,
        path: str,
        query_params: dict[str, t.Any] | None = None,
        json_data: dict[str, t.Any] | None = None,
    ) -> httpx.Response:
        """Makes a raw HTTP request without status code checking.

        This is a low-level method used by the public 'request' method.

        Args:
            method: The HTTP method (e.g., 'GET', 'POST', 'PUT', 'DELETE').
            path: The API endpoint path (relative to the base URL).
            query_params: Optional dictionary of query string parameters.
            json_data: Optional dictionary to be sent as JSON in the request body.

        Returns:
            The raw httpx.Response object received from the server.
        """
        return self._client.request(method, path, json=json_data, params=query_params)

    def request(
        self,
        method: str,
        path: str,
        query_params: dict[str, t.Any] | None = None,
        json_data: dict[str, t.Any] | None = None,
    ) -> httpx.Response:
        """Makes an HTTP request and raises an exception for error status codes.

        Wraps the internal `_request` method, adding error handling for
        non-2xx status codes. Provides specific handling for 401 Unauthorized errors.

        Args:
            method: The HTTP method (e.g., 'GET', 'POST').
            path: The API endpoint path.
            query_params: Optional dictionary of query string parameters.
            json_data: Optional dictionary to be sent as JSON in the request body.

        Returns:
            The httpx.Response object if the request was successful (2xx status).

        Raises:
            Exception: If the response status code is 401 (Unauthorized),
                       or if it's any other 4xx or 5xx error code. The error
                       message attempts to include details from the response body.
        """
        response = self._request(method, path, query_params, json_data)

        # Specific check for authentication failure
        if response.status_code == 401:
            # Consider defining a custom AuthenticationError exception
            raise Exception("Authentication failed, please check your API token.")

        try:
            # Raise an exception for 4xx or 5xx status codes
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            # Use the helper to get a detailed error message
            error_message = self._get_error_message(response)
            # Consider defining custom exception types based on status codes
            raise Exception(error_message) from e