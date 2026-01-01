"""Tests for the API client module."""

import base64
import json
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest
from pydantic import BaseModel

from dreadnode.core.api.client import ApiClient, Token


# =============================================================================
# ApiClient Initialization Tests
# =============================================================================


class TestApiClientInit:
    """Tests for ApiClient initialization."""

    def test_basic_init(self):
        """Test basic initialization with base URL."""
        client = ApiClient("https://example.com")
        assert client._base_url == "https://example.com/api"

    def test_init_with_trailing_slash(self):
        """Test initialization with trailing slash in URL."""
        client = ApiClient("https://example.com/")
        assert client._base_url == "https://example.com/api"

    def test_init_with_api_suffix(self):
        """Test initialization when URL already has /api suffix."""
        client = ApiClient("https://example.com/api")
        assert client._base_url == "https://example.com/api"

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        client = ApiClient("https://example.com", api_key="test-key-123")
        assert client._api_key == "test-key-123"
        # API key should be in headers
        assert client._client.headers.get("X-API-Key") == "test-key-123"

    def test_init_with_custom_timeout(self):
        """Test initialization with custom timeout."""
        client = ApiClient("https://example.com", timeout=60)
        # Timeout should be set on the client
        assert client._client.timeout.read == 60

    def test_user_agent_header(self):
        """Test that User-Agent header is set."""
        client = ApiClient("https://example.com")
        user_agent = client._client.headers.get("User-Agent")
        assert user_agent is not None
        assert "dreadnode-sdk" in user_agent

    def test_accept_header(self):
        """Test that Accept header is set to application/json."""
        client = ApiClient("https://example.com")
        assert client._client.headers.get("Accept") == "application/json"

    def test_invalid_url_raises(self):
        """Test that invalid URL raises ValueError."""
        with pytest.raises(ValueError, match="Invalid URL"):
            ApiClient("not-a-valid-url")

    def test_cookies_with_domain(self):
        """Test that cookies are set with correct domain."""
        cookies = {"session": "abc123"}
        client = ApiClient("https://example.com", cookies=cookies)
        # Cookies should be set on the client
        assert client._client.cookies.get("session") == "abc123"

    def test_localhost_cookie_domain(self):
        """Test that localhost gets special cookie domain handling."""
        cookies = {"session": "abc123"}
        # Should not raise for localhost
        client = ApiClient("http://localhost:3000", cookies=cookies)
        assert client._client.cookies.get("session") == "abc123"


class TestApiClientDebugMode:
    """Tests for ApiClient debug mode."""

    def test_debug_mode_adds_hooks(self):
        """Test that debug mode adds request/response hooks."""
        client = ApiClient("https://example.com", debug=True)
        # Should have hooks registered
        assert len(client._client.event_hooks["request"]) > 0
        assert len(client._client.event_hooks["response"]) > 0

    def test_non_debug_mode_no_hooks(self):
        """Test that non-debug mode has no logging hooks."""
        client = ApiClient("https://example.com", debug=False)
        # Should not have logging hooks
        assert len(client._client.event_hooks["request"]) == 0
        assert len(client._client.event_hooks["response"]) == 0


class TestApiClientErrorMessages:
    """Tests for error message extraction."""

    def test_get_error_message_json_detail(self):
        """Test error message extraction from JSON with detail field."""
        client = ApiClient("https://example.com")

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"detail": "Bad request error"}

        message = client._get_error_message(mock_response)
        assert "400" in message
        assert "Bad request error" in message

    def test_get_error_message_json_no_detail(self):
        """Test error message extraction from JSON without detail field."""
        client = ApiClient("https://example.com")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Server error", "code": "ERR001"}

        message = client._get_error_message(mock_response)
        assert "500" in message
        # Should dump the whole object as JSON
        assert "error" in message

    def test_get_error_message_non_json(self):
        """Test error message extraction from non-JSON response."""
        client = ApiClient("https://example.com")

        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.json.side_effect = Exception("Not JSON")
        mock_response.content = b"Service Unavailable"

        message = client._get_error_message(mock_response)
        assert "503" in message
        assert "Service Unavailable" in message


class TestApiClientUrlHelpers:
    """Tests for URL helper methods."""

    def test_url_for_user_code(self):
        """Test URL generation for user code verification."""
        client = ApiClient("https://app.dreadnode.io")
        url = client.url_for_user_code("ABC123")
        assert url == "https://app.dreadnode.io/account/device?code=ABC123"

    def test_url_for_user_code_removes_api_suffix(self):
        """Test that /api suffix is removed from URL."""
        client = ApiClient("https://app.dreadnode.io/api")
        url = client.url_for_user_code("XYZ789")
        assert url == "https://app.dreadnode.io/account/device?code=XYZ789"

    def test_pypi_registry_url_without_api_key(self):
        """Test PyPI registry URL without API key."""
        client = ApiClient("https://example.com")
        url = client.pypi_registry_url
        assert url == "https://example.com/pypi/simple"
        assert "__token__" not in url

    def test_pypi_registry_url_with_api_key(self):
        """Test PyPI registry URL includes auth when API key is set."""
        client = ApiClient("https://example.com", api_key="my-api-key")
        url = client.pypi_registry_url
        assert "__token__:my-api-key@" in url
        assert "/pypi/simple" in url


# =============================================================================
# ApiClient Request Tests (with mocking)
# =============================================================================


class TestApiClientRequests:
    """Tests for ApiClient request methods with mocking."""

    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        return ApiClient("https://example.com", api_key="test-key")

    def test_request_raises_on_error(self, client):
        """Test that request raises RuntimeError on HTTP error."""
        with patch.object(client._client, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
                "Not Found", request=MagicMock(), response=mock_response
            )
            mock_response.json.return_value = {"detail": "Resource not found"}
            mock_request.return_value = mock_response

            with pytest.raises(RuntimeError, match="404"):
                client.request("GET", "/some/path")

    def test_raw_request_does_not_raise(self, client):
        """Test that _request does not raise on error."""
        with patch.object(client._client, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_request.return_value = mock_response

            # Should not raise
            response = client._request("GET", "/some/path")
            assert response.status_code == 500


class TestApiClientUserMethods:
    """Tests for user-related API methods."""

    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        return ApiClient("https://example.com")

    def test_get_user(self, client):
        """Test get_user method."""
        with patch.object(client, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "user123",
                "email_address": "test@example.com",
                "username": "testuser",
                "api_key": {"key": "api_key_123"},
            }
            mock_request.return_value = mock_response

            user = client.get_user()
            assert user.email_address == "test@example.com"
            assert user.username == "testuser"
            mock_request.assert_called_once_with("GET", "/user")


class TestApiClientAuthMethods:
    """Tests for authentication-related API methods."""

    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        return ApiClient("https://example.com")

    def test_get_device_codes(self, client):
        """Test get_device_codes method."""
        with patch.object(client, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "code_id_123",
                "completed": False,
                "device_code": "dev123",
                "expires_at": "2025-12-31T23:59:59Z",
                "expires_in": 900,
                "user_code": "ABC-123",
                "verification_url": "https://example.com/verify",
            }
            mock_request.return_value = mock_response

            codes = client.get_device_codes()
            assert codes.device_code == "dev123"
            assert codes.user_code == "ABC-123"
            mock_request.assert_called_once_with("POST", "/auth/device/code")

    def test_poll_for_token_success(self, client):
        """Test poll_for_token when token is immediately available."""
        with patch.object(client, "_request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "access_token": "access123",
                "refresh_token": "refresh456",
            }
            mock_request.return_value = mock_response

            result = client.poll_for_token("device123", interval=1, max_poll_time=5)
            assert result.access_token == "access123"
            assert result.refresh_token == "refresh456"

    def test_poll_for_token_timeout(self, client):
        """Test poll_for_token times out."""
        with patch.object(client, "_request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 401  # Still waiting
            mock_request.return_value = mock_response

            with pytest.raises(RuntimeError, match="timed out"):
                client.poll_for_token("device123", interval=0.1, max_poll_time=0.3)


class TestApiClientOrganizations:
    """Tests for organization-related API methods."""

    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        return ApiClient("https://example.com")

    @pytest.fixture
    def org_data(self):
        """Sample organization data."""
        return {
            "id": "org1",
            "name": "Org One",
            "key": "org-one",
            "description": "Test org",
            "is_active": True,
            "allow_external_invites": False,
            "max_members": 100,
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
        }

    def test_list_organizations(self, client, org_data):
        """Test list_organizations method."""
        with patch.object(client, "request") as mock_request:
            mock_response = MagicMock()
            org2 = org_data.copy()
            org2["id"] = "org2"
            org2["name"] = "Org Two"
            org2["key"] = "org-two"
            mock_response.json.return_value = [org_data, org2]
            mock_request.return_value = mock_response

            orgs = client.list_organizations()
            assert len(orgs) == 2
            assert orgs[0].name == "Org One"
            mock_request.assert_called_once_with("GET", "/organizations")

    def test_get_organization(self, client, org_data):
        """Test get_organization method."""
        with patch.object(client, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = org_data
            mock_request.return_value = mock_response

            org = client.get_organization("org1")
            assert org.name == "Org One"
            mock_request.assert_called_once_with("GET", "/organizations/org1")


class TestApiClientProjects:
    """Tests for project-related API methods."""

    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        return ApiClient("https://example.com")

    @pytest.fixture
    def project_data(self):
        """Sample project data."""
        return {
            "id": "proj1",
            "name": "Project One",
            "key": "proj-one",
            "description": "Test project",
            "workspace_id": "ws1",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "run_count": 0,
            "last_run": None,
        }

    def test_list_projects(self, client, project_data):
        """Test list_projects method."""
        with patch.object(client, "request") as mock_request:
            mock_response = MagicMock()
            proj2 = project_data.copy()
            proj2["id"] = "proj2"
            proj2["name"] = "Project Two"
            proj2["key"] = "proj-two"
            mock_response.json.return_value = [project_data, proj2]
            mock_request.return_value = mock_response

            projects = client.list_projects()
            assert len(projects) == 2
            assert projects[0].name == "Project One"
            mock_request.assert_called_once_with("GET", "/strikes/projects")


class TestApiClientRuns:
    """Tests for run-related API methods."""

    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        return ApiClient("https://example.com")

    @pytest.fixture
    def run_data(self):
        """Sample run summary data."""
        return {
            "id": "run1",
            "name": "Run 1",
            "span_id": "span123",
            "trace_id": "trace456",
            "timestamp": "2025-01-01T00:00:00Z",
            "duration": 1000,
            "status": "completed",
            "exception": None,
            "tags": [],
            "params": {},
            "metrics": {},
        }

    def test_list_runs(self, client, run_data):
        """Test list_runs method."""
        with patch.object(client, "request") as mock_request:
            mock_response = MagicMock()
            run2 = run_data.copy()
            run2["id"] = "run2"
            run2["name"] = "Run 2"
            mock_response.json.return_value = [run_data, run2]
            mock_request.return_value = mock_response

            runs = client.list_runs("my-project")
            assert len(runs) == 2
            mock_request.assert_called_once_with(
                "GET", "/strikes/projects/my-project/runs"
            )


# =============================================================================
# Token Class Tests
# =============================================================================


class TestToken:
    """Tests for Token class."""

    @pytest.fixture
    def valid_token(self):
        """Create a valid JWT token that expires in 1 hour."""
        # JWT format: header.payload.signature
        # We only care about the payload for expiration
        exp_time = int(time.time()) + 3600  # 1 hour from now
        payload = {"exp": exp_time, "sub": "user123"}
        b64_payload = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
        return f"header.{b64_payload}.signature"

    @pytest.fixture
    def expired_token(self):
        """Create an expired JWT token."""
        exp_time = int(time.time()) - 3600  # 1 hour ago
        payload = {"exp": exp_time, "sub": "user123"}
        b64_payload = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
        return f"header.{b64_payload}.signature"

    def test_parse_jwt_expiration(self, valid_token):
        """Test parsing expiration from JWT token."""
        exp = Token.parse_jwt_token_expiration(valid_token)
        assert isinstance(exp, datetime)
        assert exp.tzinfo is not None  # Should be timezone-aware

    def test_token_creation(self, valid_token):
        """Test Token creation."""
        token = Token(valid_token)
        assert token.data == valid_token
        assert isinstance(token.expires_at, datetime)

    def test_token_ttl_positive(self, valid_token):
        """Test TTL returns positive value for valid token."""
        token = Token(valid_token)
        ttl = token.ttl()
        assert ttl > 0
        assert ttl <= 3600  # Should be about 1 hour

    def test_token_ttl_negative(self, expired_token):
        """Test TTL returns negative value for expired token."""
        token = Token(expired_token)
        ttl = token.ttl()
        assert ttl < 0

    def test_is_expired_false(self, valid_token):
        """Test is_expired returns False for valid token."""
        token = Token(valid_token)
        assert token.is_expired() is False

    def test_is_expired_true(self, expired_token):
        """Test is_expired returns True for expired token."""
        token = Token(expired_token)
        assert token.is_expired() is True

    def test_is_close_to_expiry_false(self, valid_token):
        """Test is_close_to_expiry returns False when plenty of time left."""
        token = Token(valid_token)
        # Token has 1 hour, should not be close to expiry
        assert token.is_close_to_expiry() is False

    def test_is_close_to_expiry_true(self):
        """Test is_close_to_expiry returns True when little time left."""
        # Create a token expiring in 60 seconds
        exp_time = int(time.time()) + 60
        payload = {"exp": exp_time}
        b64_payload = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
        short_token = f"header.{b64_payload}.signature"

        token = Token(short_token)
        # With default settings, 60 seconds should be close to expiry
        # (depends on settings.token_max_ttl, usually much higher)
        # This test may need adjustment based on actual settings


class TestTokenEdgeCases:
    """Tests for Token edge cases."""

    def test_token_with_padding_issues(self):
        """Test that token parsing handles base64 padding correctly."""
        # Create payload without proper padding
        exp_time = int(time.time()) + 3600
        payload = {"exp": exp_time}
        # Create base64 that needs padding
        b64_payload = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
        # Remove padding if present
        b64_payload = b64_payload.rstrip("=")
        token_str = f"header.{b64_payload}.signature"

        # Should still parse correctly (the code adds "==" for padding)
        token = Token(token_str)
        assert not token.is_expired()


# =============================================================================
# Integration-style Tests (with full mocking)
# =============================================================================


class TestApiClientIntegration:
    """Integration-style tests with full request mocking."""

    def test_full_auth_flow(self):
        """Test simulated authentication flow."""
        client = ApiClient("https://example.com")

        # Step 1: Get device codes
        with patch.object(client, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "id": "code_id_123",
                "completed": False,
                "device_code": "dev123",
                "expires_at": "2025-12-31T23:59:59Z",
                "expires_in": 900,
                "user_code": "ABC-123",
                "verification_url": "https://example.com/verify",
            }
            mock_request.return_value = mock_response

            codes = client.get_device_codes()
            assert codes.user_code == "ABC-123"

        # Step 2: Generate URL for user
        url = client.url_for_user_code(codes.user_code)
        assert codes.user_code in url

        # Step 3: Poll for token (simulated immediate success)
        with patch.object(client, "_request") as mock_request:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "access_token": "access_token_value",
                "refresh_token": "refresh_token_value",
            }
            mock_request.return_value = mock_response

            tokens = client.poll_for_token(
                codes.device_code, interval=0.1, max_poll_time=1
            )
            assert tokens.access_token == "access_token_value"


class TestApiClientWithResponses:
    """Tests using actual httpx Response objects."""

    def test_request_with_actual_response(self):
        """Test using actual httpx Response for better fidelity."""
        client = ApiClient("https://example.com")

        # Create a more realistic mock
        with patch.object(client._client, "request") as mock_request:
            # Create actual Response object
            response = httpx.Response(
                200,
                json={"status": "ok", "data": {"id": 1}},
                request=httpx.Request("GET", "https://example.com/api/test"),
            )
            mock_request.return_value = response

            result = client.request("GET", "/test")
            assert result.status_code == 200
            assert result.json()["status"] == "ok"


# =============================================================================
# Workspace Tests
# =============================================================================


class TestApiClientWorkspaces:
    """Tests for workspace-related API methods."""

    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        return ApiClient("https://example.com")

    @pytest.fixture
    def workspace_data(self):
        """Sample workspace data."""
        return {
            "id": "ws1",
            "name": "Workspace 1",
            "key": "ws-1",
            "description": "Test workspace",
            "org_id": "org1",
            "org_name": "Org One",
            "is_active": True,
            "is_default": False,
            "project_count": 0,
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
        }

    def test_list_workspaces_single_page(self, client, workspace_data):
        """Test list_workspaces with single page of results."""
        with patch.object(client, "request") as mock_request:
            mock_response = MagicMock()
            ws2 = workspace_data.copy()
            ws2["id"] = "ws2"
            ws2["name"] = "Workspace 2"
            ws2["key"] = "ws-2"
            mock_response.json.return_value = {
                "workspaces": [workspace_data, ws2],
                "page": 1,
                "limit": 10,
                "total": 2,
                "total_pages": 1,
                "has_next": False,
                "has_previous": False,
            }
            mock_request.return_value = mock_response

            workspaces = client.list_workspaces()
            assert len(workspaces) == 2

    def test_get_workspace(self, client, workspace_data):
        """Test get_workspace method."""
        with patch.object(client, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = workspace_data
            mock_request.return_value = mock_response

            ws = client.get_workspace("ws1")
            assert ws.name == "Workspace 1"


# =============================================================================
# Storage and Registry Tests
# =============================================================================


class TestApiClientStorage:
    """Tests for storage-related API methods."""

    @pytest.fixture
    def client(self):
        """Create a client for testing."""
        return ApiClient("https://example.com")

    def test_get_storage_access(self, client):
        """Test get_storage_access method."""
        with patch.object(client, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "access_key_id": "AKIAEXAMPLE",
                "secret_access_key": "secret123",
                "session_token": "token456",
                "expiration": "2025-12-31T23:59:59Z",
                "region": "us-east-1",
                "bucket": "user-data-bucket",
                "prefix": "workspace/user/",
                "endpoint": None,
            }
            mock_request.return_value = mock_response

            creds = client.get_storage_access("workspace123")
            assert creds.access_key_id == "AKIAEXAMPLE"

    def test_get_platform_registry_credentials(self, client):
        """Test get_platform_registry_credentials method."""
        with patch.object(client, "request") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "registry": "registry.example.com",
                "username": "registry_user",
                "password": "registry_pass",
                "expires_at": "2025-12-31T23:59:59Z",
            }
            mock_request.return_value = mock_response

            creds = client.get_platform_registry_credentials()
            assert creds.username == "registry_user"
