"""Tests for PreFlightValidator class."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dreadnode.airgap.validator import PreFlightValidator, ValidationError


class TestPreFlightValidator:
    """Test suite for PreFlightValidator."""

    def test_init(self):
        """Test validator initialization."""
        validator = PreFlightValidator()
        assert validator.errors == []
        assert validator.warnings == []

    @patch("shutil.disk_usage")
    def test_check_disk_space_sufficient(self, mock_disk_usage):
        """Test disk space check with sufficient space."""
        mock_disk_usage.return_value = MagicMock(free=30 * 1024**3)  # 30GB

        validator = PreFlightValidator()
        validator.check_disk_space()

        assert len(validator.errors) == 0

    @patch("shutil.disk_usage")
    def test_check_disk_space_insufficient(self, mock_disk_usage):
        """Test disk space check with insufficient space."""
        mock_disk_usage.return_value = MagicMock(free=10 * 1024**3)  # 10GB

        validator = PreFlightValidator()
        validator.check_disk_space()

        assert len(validator.errors) == 1
        assert "Insufficient disk space" in validator.errors[0]

    @patch("shutil.which")
    def test_check_required_tools_all_present(self, mock_which):
        """Test required tools check when all tools are present."""
        mock_which.return_value = "/usr/local/bin/tool"

        validator = PreFlightValidator()
        validator.check_required_tools()

        assert len(validator.errors) == 0

    @patch("shutil.which")
    def test_check_required_tools_missing(self, mock_which):
        """Test required tools check when tools are missing."""
        mock_which.return_value = None

        validator = PreFlightValidator()
        validator.check_required_tools()

        assert len(validator.errors) >= 2  # zarf and kubectl

    def test_check_bundle_exists_valid(self, tmp_path):
        """Test bundle exists check with valid bundle."""
        bundle = tmp_path / "test.tar.zst"
        bundle.write_bytes(b"test data" * 1024 * 1024)  # 9MB

        validator = PreFlightValidator()
        validator.check_bundle_exists(bundle)

        assert len(validator.errors) == 0

    def test_check_bundle_exists_missing(self, tmp_path):
        """Test bundle exists check with missing bundle."""
        bundle = tmp_path / "nonexistent.tar.zst"

        validator = PreFlightValidator()
        validator.check_bundle_exists(bundle)

        assert len(validator.errors) == 1
        assert "Bundle file not found" in validator.errors[0]

    def test_check_bundle_exists_too_small(self, tmp_path):
        """Test bundle exists check with suspiciously small bundle."""
        bundle = tmp_path / "tiny.tar.zst"
        bundle.write_bytes(b"tiny")

        validator = PreFlightValidator()
        validator.check_bundle_exists(bundle)

        assert len(validator.warnings) == 1
        assert "Bundle file seems small" in validator.warnings[0]

    def test_check_ecr_connectivity(self):
        """Test ECR connectivity check."""
        mock_ecr = MagicMock()
        mock_ecr.verify_connectivity.return_value = True

        validator = PreFlightValidator()
        validator.check_ecr_connectivity(mock_ecr)

        assert len(validator.errors) == 0

    def test_check_ecr_connectivity_failure(self):
        """Test ECR connectivity check failure."""
        mock_ecr = MagicMock()
        mock_ecr.verify_connectivity.return_value = False
        mock_ecr.registry_url = "123456.dkr.ecr.us-east-1.amazonaws.com"

        validator = PreFlightValidator()
        validator.check_ecr_connectivity(mock_ecr)

        assert len(validator.errors) == 1
        assert "Cannot connect to ECR" in validator.errors[0]

    def test_check_ecr_credentials(self):
        """Test ECR credentials check."""
        mock_ecr = MagicMock()
        mock_ecr.verify_credentials.return_value = (True, None)

        validator = PreFlightValidator()
        validator.check_ecr_credentials(mock_ecr)

        assert len(validator.errors) == 0

    def test_check_ecr_credentials_failure(self):
        """Test ECR credentials check failure."""
        mock_ecr = MagicMock()
        mock_ecr.verify_credentials.return_value = (False, "Invalid credentials")

        validator = PreFlightValidator()
        validator.check_ecr_credentials(mock_ecr)

        assert len(validator.errors) == 1
        assert "ECR credentials validation failed" in validator.errors[0]

    def test_validate_all_success(self):
        """Test validate_all with all checks passing."""
        validator = PreFlightValidator()

        # Mock all validation methods to succeed
        with patch.multiple(
            validator,
            check_disk_space=MagicMock(),
            check_required_tools=MagicMock(),
            check_kubectl_available=MagicMock(),
            check_kubernetes_connectivity=MagicMock(),
        ):
            validator.validate_all(skip_k8s=False)

            # Should not raise exception
            assert len(validator.errors) == 0

    def test_validate_all_with_errors(self):
        """Test validate_all with validation errors."""
        validator = PreFlightValidator()
        validator.errors.append("Test error")

        with pytest.raises(ValidationError, match="Test error"):
            validator.validate_all()

    def test_validate_zarf_package(self, tmp_path):
        """Test Zarf package validation."""
        validator = PreFlightValidator()

        # Valid package name
        valid_package = tmp_path / "test-package.tar.zst"
        valid_package.touch()
        assert validator.validate_zarf_package(valid_package) is True

        # Invalid package name
        invalid_package = tmp_path / "test-package.zip"
        invalid_package.touch()
        assert validator.validate_zarf_package(invalid_package) is False
