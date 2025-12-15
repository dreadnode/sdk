"""Tests for ECRHelper class."""

from unittest.mock import MagicMock, patch

import pytest

from dreadnode.airgap.ecr_helper import ECRHelper


@pytest.fixture
def mock_boto3():
    """Mock boto3 for testing."""
    with patch("dreadnode.airgap.ecr_helper.boto3") as mock:
        mock_client = MagicMock()
        mock.client.return_value = mock_client
        yield mock, mock_client


class TestECRHelper:
    """Test suite for ECRHelper."""

    def test_init_extracts_account_id_and_region(self, mock_boto3):
        """Test that initialization extracts account ID and region."""
        _mock_boto, _ = mock_boto3

        helper = ECRHelper("123456789.dkr.ecr.us-east-1.amazonaws.com")

        assert helper.account_id == "123456789"
        assert helper.region == "us-east-1"
        assert helper.registry_url == "123456789.dkr.ecr.us-east-1.amazonaws.com"

    def test_init_invalid_registry_url(self, mock_boto3):  # noqa: ARG002
        """Test that invalid registry URL raises error."""
        with pytest.raises(ValueError, match="Invalid ECR registry URL"):
            ECRHelper("invalid-url")

    def test_verify_connectivity_success(self, mock_boto3):
        """Test successful connectivity verification."""
        _, mock_client = mock_boto3
        mock_client.describe_repositories.return_value = {"repositories": []}

        helper = ECRHelper("123456789.dkr.ecr.us-east-1.amazonaws.com")
        assert helper.verify_connectivity() is True

    def test_verify_connectivity_failure(self, mock_boto3):
        """Test connectivity verification failure."""
        _, mock_client = mock_boto3
        mock_client.describe_repositories.side_effect = Exception("Connection error")

        helper = ECRHelper("123456789.dkr.ecr.us-east-1.amazonaws.com")
        assert helper.verify_connectivity() is False

    def test_list_repositories(self, mock_boto3):
        """Test listing ECR repositories."""
        _, mock_client = mock_boto3
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"repositories": [{"repositoryName": "repo1"}, {"repositoryName": "repo2"}]}
        ]
        mock_client.get_paginator.return_value = mock_paginator

        helper = ECRHelper("123456789.dkr.ecr.us-east-1.amazonaws.com")
        repos = helper.list_repositories()

        assert repos == ["repo1", "repo2"]

    def test_extract_repo_name(self, mock_boto3):  # noqa: ARG002
        """Test extracting repository name from image string."""
        helper = ECRHelper("123456789.dkr.ecr.us-east-1.amazonaws.com")

        # Test various formats
        assert helper._extract_repo_name("platform-api:v1.0.0") == "platform-api"
        assert helper._extract_repo_name("platform-api") == "platform-api"
        assert helper._extract_repo_name("org/platform-api:latest") == "org/platform-api"
        assert helper._extract_repo_name("platform-api@sha256:abc123") == "platform-api"

    def test_ensure_repositories_creates_missing(self, mock_boto3):
        """Test that ensure_repositories creates missing repos."""
        _, mock_client = mock_boto3

        # Mock existing repos
        mock_paginator = MagicMock()
        mock_paginator.paginate.return_value = [
            {"repositories": [{"repositoryName": "existing-repo"}]}
        ]
        mock_client.get_paginator.return_value = mock_paginator

        helper = ECRHelper("123456789.dkr.ecr.us-east-1.amazonaws.com")

        images = ["existing-repo:v1", "new-repo:v1", "another-new-repo:v2"]
        created = helper.ensure_repositories(images)

        assert len(created) == 2
        assert "new-repo" in created
        assert "another-new-repo" in created
        assert mock_client.create_repository.call_count == 2

    def test_get_repository_uri(self, mock_boto3):  # noqa: ARG002
        """Test getting repository URI."""
        helper = ECRHelper("123456789.dkr.ecr.us-east-1.amazonaws.com")

        uri = helper.get_repository_uri("platform-api")
        assert uri == "123456789.dkr.ecr.us-east-1.amazonaws.com/platform-api"
