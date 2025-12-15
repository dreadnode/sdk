"""Tests for ZarfWrapper class."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dreadnode.airgap.zarf_wrapper import ZarfWrapper


class TestZarfWrapper:
    """Test suite for ZarfWrapper."""

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_init_verifies_zarf_available(self, mock_run, mock_which):
        """Test that initialization verifies Zarf is available."""
        mock_which.return_value = "/usr/local/bin/zarf"
        mock_run.return_value = MagicMock(stdout="v0.32.0", returncode=0)

        wrapper = ZarfWrapper()
        assert wrapper.zarf_binary == "zarf"
        mock_which.assert_called_once_with("zarf")
        mock_run.assert_called_once()

    @patch("shutil.which")
    def test_init_raises_if_zarf_not_found(self, mock_which):
        """Test that initialization raises error if Zarf not found."""
        mock_which.return_value = None

        with pytest.raises(RuntimeError, match="Zarf binary.*not found"):
            ZarfWrapper()

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_create_package_success(self, mock_run, mock_which):
        """Test successful package creation."""
        mock_which.return_value = "/usr/local/bin/zarf"
        mock_run.side_effect = [
            MagicMock(stdout="v0.32.0", returncode=0),  # version check
            MagicMock(stdout="Package created", returncode=0),  # create
        ]

        wrapper = ZarfWrapper()

        # Mock the _find_package method
        with patch.object(wrapper, "_find_package") as mock_find:
            mock_find.return_value = Path("/tmp/test-package.tar.zst")

            result = wrapper.create_package(
                source_dir=Path("/tmp/source"),
                output_dir=Path("/tmp/output"),
                version="v1.0.0",
            )

            assert result == Path("/tmp/test-package.tar.zst")
            assert mock_run.call_count == 2

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_create_package_failure(self, mock_run, mock_which):
        """Test package creation failure."""
        mock_which.return_value = "/usr/local/bin/zarf"
        mock_run.side_effect = [
            MagicMock(stdout="v0.32.0", returncode=0),  # version check
            subprocess.CalledProcessError(1, "zarf", stderr="Error creating package"),
        ]

        wrapper = ZarfWrapper()

        with pytest.raises(RuntimeError, match="Zarf package creation failed"):
            wrapper.create_package(
                source_dir=Path("/tmp/source"),
                output_dir=Path("/tmp/output"),
            )

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_inspect_package_success(self, mock_run, mock_which, tmp_path):
        """Test successful package inspection."""
        mock_which.return_value = "/usr/local/bin/zarf"
        mock_run.side_effect = [
            MagicMock(stdout="v0.32.0", returncode=0),  # version check
            MagicMock(
                stdout='{"name": "test-package", "version": "v1.0.0"}', returncode=0
            ),  # inspect
        ]

        wrapper = ZarfWrapper()

        package_path = tmp_path / "test-package.tar.zst"
        package_path.touch()

        result = wrapper.inspect_package(package_path)

        assert result == {"name": "test-package", "version": "v1.0.0"}
        assert mock_run.call_count == 2

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_deploy_package_success(self, mock_run, mock_which, tmp_path):
        """Test successful package deployment."""
        mock_which.return_value = "/usr/local/bin/zarf"
        mock_run.side_effect = [
            MagicMock(stdout="v0.32.0", returncode=0),  # version check
            MagicMock(stdout="Package deployed", returncode=0),  # deploy
        ]

        wrapper = ZarfWrapper()

        package_path = tmp_path / "test-package.tar.zst"
        package_path.touch()

        wrapper.deploy_package(package_path, components=["platform"])

        assert mock_run.call_count == 2
        deploy_call = mock_run.call_args_list[1]
        assert "--components=platform" in deploy_call[0][0]

    @patch("shutil.which")
    @patch("subprocess.run")
    def test_list_packages(self, mock_run, mock_which, tmp_path):
        """Test listing packages in directory."""
        mock_which.return_value = "/usr/local/bin/zarf"
        mock_run.return_value = MagicMock(stdout="v0.32.0", returncode=0)

        wrapper = ZarfWrapper()

        # Create test packages
        (tmp_path / "package1.tar.zst").touch()
        (tmp_path / "package2.tar.zst").touch()
        (tmp_path / "other.txt").touch()

        packages = wrapper.list_packages(tmp_path)

        assert len(packages) == 2
        assert all(p.suffix == ".zst" for p in packages)
