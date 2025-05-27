"""
Tests for the main dreadnode module and configuration functionality.
"""

import tempfile
from pathlib import Path

import dreadnode


def test_default_instance():
    """Test that the default instance is properly initialized."""
    assert dreadnode.DEFAULT_INSTANCE is not None
    assert isinstance(dreadnode.DEFAULT_INSTANCE, dreadnode.Dreadnode)


def test_version():
    """Test that the version is properly exported."""
    assert dreadnode.__version__ is not None
    assert isinstance(dreadnode.__version__, str)


class TestDreadnodeConfig:
    """Tests for the Dreadnode configuration."""

    def test_configure_local_dir(self, test_instance):
        """Test configuring the local directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_instance.configure(local_dir=temp_dir)
            assert test_instance.config.local_dir == Path(temp_dir)

            # Check that directories are created
            assert Path(temp_dir).exists()
            assert Path(temp_dir, "telemetry").exists()

    def test_configure_project(self, test_instance):
        """Test configuring the project."""
        test_instance.configure(project="test-project")
        assert test_instance.config.project == "test-project"

    def test_env_vars_config(self, mock_env_vars):
        """Test that environment variables are properly used."""
        instance = dreadnode.Dreadnode()
        instance.configure()
        assert instance.config.project == "test-env-project"
        assert str(instance.config.local_dir).endswith("test-env-project")
        assert instance.config.api_key == "test-api-key"
        instance.shutdown()

    def test_config_precedence(self, mock_env_vars):
        """Test that explicit config overrides environment variables."""
        instance = dreadnode.Dreadnode()
        instance.configure(project="explicit-project")
        assert instance.config.project == "explicit-project"  # Should override env var
        assert instance.config.api_key == "test-api-key"  # Should use env var
        instance.shutdown()

    def test_shutdown(self, test_instance):
        """Test that shutdown properly closes resources."""
        test_instance.shutdown()
        # Test that we can configure again after shutdown
        with tempfile.TemporaryDirectory() as temp_dir:
            test_instance.configure(local_dir=temp_dir)
            assert test_instance.config.local_dir == Path(temp_dir)
