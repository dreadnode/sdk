import tempfile
import typing as t
from pathlib import Path

import pytest

import dreadnode


@pytest.fixture
def tmp_path_factory() -> t.Generator[Path]:
    """Create a temporary directory that is cleaned up after the test."""
    temp_dir = tempfile.TemporaryDirectory()
    yield Path(temp_dir.name)
    temp_dir.cleanup()


@pytest.fixture
def test_instance():
    """Create a clean test instance of dreadnode."""
    instance = dreadnode.Dreadnode()
    temp_dir = tempfile.TemporaryDirectory()
    instance.configure(local_dir=temp_dir.name)
    yield instance
    instance.shutdown()
    temp_dir.cleanup()


@pytest.fixture
def configured_instance():
    """Create a preconfigured test instance."""
    temp_dir = tempfile.TemporaryDirectory()
    instance = dreadnode.Dreadnode()
    instance.configure(
        project="test-project",
        local_dir=temp_dir.name,
    )
    yield instance
    instance.shutdown()
    temp_dir.cleanup()


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Setup environment variables for testing."""
    temp_dir = tempfile.TemporaryDirectory()
    monkeypatch.setenv("DREADNODE_PROJECT", "test-env-project")
    monkeypatch.setenv("DREADNODE_LOCAL_DIR", temp_dir.name)
    monkeypatch.setenv("DREADNODE_API_KEY", "test-api-key")
    yield
    temp_dir.cleanup()
