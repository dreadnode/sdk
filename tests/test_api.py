"""
Tests for the API client functionality.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from dreadnode.api.client import ApiClient
from dreadnode.api.models import Project, Run, Task


# Mock responses for API calls
@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client for testing API calls."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_client.request.return_value = mock_response
    return mock_client


@pytest.fixture
def api_client(mock_httpx_client):
    """Create an API client with mocked httpx client."""
    with patch("httpx.Client", return_value=mock_httpx_client):
        client = ApiClient(base_url="https://api.example.com", api_key="test-api-key", debug=True)
        yield client


def test_api_client_initialization():
    """Test API client initialization."""
    client = ApiClient(base_url="https://api.example.com", api_key="test-api-key")

    # Check that the base URL is properly formatted
    assert client._base_url == "https://api.example.com/api"

    # Check that the client has the correct headers
    assert "X-API-Key" in client._client.headers
    assert client._client.headers["X-API-Key"] == "test-api-key"
    assert "User-Agent" in client._client.headers
    assert "dreadnode-sdk" in client._client.headers["User-Agent"]


def test_get_projects(api_client, mock_httpx_client):
    """Test getting projects from the API."""
    # Setup mock response
    mock_response = mock_httpx_client.request.return_value
    mock_response.json.return_value = {
        "data": [
            {"id": "proj-1", "name": "Test Project 1"},
            {"id": "proj-2", "name": "Test Project 2"},
        ]
    }

    # Call the API
    projects = api_client.get_projects()

    # Verify request was made correctly
    mock_httpx_client.request.assert_called_with("GET", "/projects", params={})

    # Verify response parsing
    assert len(projects) == 2
    assert all(isinstance(proj, Project) for proj in projects)
    assert projects[0].id == "proj-1"
    assert projects[1].name == "Test Project 2"


def test_get_project(api_client, mock_httpx_client):
    """Test getting a single project by ID."""
    # Setup mock response
    mock_response = mock_httpx_client.request.return_value
    mock_response.json.return_value = {"id": "proj-1", "name": "Test Project"}

    # Call the API
    project = api_client.get_project("proj-1")

    # Verify request was made correctly
    mock_httpx_client.request.assert_called_with("GET", "/projects/proj-1")

    # Verify response parsing
    assert isinstance(project, Project)
    assert project.id == "proj-1"
    assert project.name == "Test Project"


def test_get_runs(api_client, mock_httpx_client):
    """Test getting runs for a project."""
    # Setup mock response
    mock_response = mock_httpx_client.request.return_value
    mock_response.json.return_value = {
        "data": [
            {"id": "run-1", "name": "Test Run 1", "project_id": "proj-1"},
            {"id": "run-2", "name": "Test Run 2", "project_id": "proj-1"},
        ]
    }

    # Call the API
    runs = api_client.get_runs(project_id="proj-1")

    # Verify request was made correctly
    mock_httpx_client.request.assert_called_with("GET", "/runs", params={"project_id": "proj-1"})

    # Verify response parsing
    assert len(runs) == 2
    assert all(isinstance(run, Run) for run in runs)
    assert runs[0].id == "run-1"
    assert runs[1].name == "Test Run 2"
    assert all(run.project_id == "proj-1" for run in runs)


def test_post_runs(api_client, mock_httpx_client):
    """Test creating a new run."""
    # Setup mock response
    mock_response = mock_httpx_client.request.return_value
    mock_response.json.return_value = {"id": "run-new", "name": "New Run", "project_id": "proj-1"}

    # Call the API
    run = api_client.post_run(project_id="proj-1", name="New Run", tags={"env": "test"})

    # Verify request was made correctly
    mock_httpx_client.request.assert_called_with(
        "POST", "/runs", json={"project_id": "proj-1", "name": "New Run", "tags": {"env": "test"}}
    )

    # Verify response parsing
    assert isinstance(run, Run)
    assert run.id == "run-new"
    assert run.name == "New Run"
    assert run.project_id == "proj-1"


def test_get_tasks(api_client, mock_httpx_client):
    """Test getting tasks for a run."""
    # Setup mock response
    mock_response = mock_httpx_client.request.return_value
    mock_response.json.return_value = {
        "data": [
            {"id": "task-1", "name": "Test Task 1", "run_id": "run-1"},
            {"id": "task-2", "name": "Test Task 2", "run_id": "run-1"},
        ]
    }

    # Call the API
    tasks = api_client.get_tasks(run_id="run-1")

    # Verify request was made correctly
    mock_httpx_client.request.assert_called_with("GET", "/tasks", params={"run_id": "run-1"})

    # Verify response parsing
    assert len(tasks) == 2
    assert all(isinstance(task, Task) for task in tasks)
    assert tasks[0].id == "task-1"
    assert tasks[1].name == "Test Task 2"
    assert all(task.run_id == "run-1" for task in tasks)


def test_error_handling(api_client, mock_httpx_client):
    """Test error handling in API client."""
    # Setup mock error response
    mock_response = mock_httpx_client.request.return_value
    mock_response.status_code = 404
    mock_response.json.return_value = {"error": "Project not found"}
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Not Found", request=httpx.Request("GET", "/projects/invalid"), response=mock_response
    )

    # Call the API and expect an exception
    with pytest.raises(httpx.HTTPStatusError):
        api_client.get_project("invalid")

    # Verify request was made correctly
    mock_httpx_client.request.assert_called_with("GET", "/projects/invalid")
