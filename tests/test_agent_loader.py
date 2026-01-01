"""Tests for agent loading and storage."""

from unittest.mock import MagicMock, patch

import pytest

from dreadnode.agents.loader import AgentPackage
from dreadnode.agents.local import LocalAgent
from dreadnode.core.agents import Agent
from dreadnode.core.packaging.manifest import AgentManifest
from dreadnode.core.storage.storage import Storage

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def temp_storage(tmp_path):
    """Create a temporary storage instance."""
    cache_dir = tmp_path / ".dreadnode"
    cache_dir.mkdir()
    return Storage(cache=cache_dir)


@pytest.fixture
def mock_agent():
    """Create a mock Agent for testing."""
    agent = MagicMock(spec=Agent)
    agent.name = "test-agent"
    agent.description = "A test agent"
    agent.model = "gpt-4"
    agent.instructions = "You are a helpful assistant."
    agent.max_steps = 10
    agent.tags = ["test"]
    agent.all_tools = []
    return agent


@pytest.fixture
def real_agent():
    """Create a real minimal Agent for testing."""
    return Agent(
        name="test-agent",
        description="A test agent",
        model="gpt-4",
        instructions="You are a helpful assistant.",
        max_steps=5,
    )


# ==============================================================================
# LocalAgent Tests
# ==============================================================================


class TestLocalAgentFromAgent:
    """Tests for LocalAgent.from_agent()."""

    def test_from_agent_creates_manifest(self, real_agent, temp_storage):
        """Test that from_agent creates proper manifest."""
        local_agent = LocalAgent.from_agent(
            real_agent,
            "test-agent",
            temp_storage,
        )

        assert local_agent.name == "test-agent"
        assert local_agent.version == "0.1.0"
        assert "config.json" in local_agent.manifest.artifacts

    def test_from_agent_custom_version(self, real_agent, temp_storage):
        """Test from_agent with custom version."""
        local_agent = LocalAgent.from_agent(
            real_agent,
            "test-agent",
            temp_storage,
            version="1.0.0",
        )

        assert local_agent.version == "1.0.0"

    def test_from_agent_stores_config(self, real_agent, temp_storage):
        """Test that agent config is properly stored."""
        local_agent = LocalAgent.from_agent(
            real_agent,
            "test-agent",
            temp_storage,
        )

        config = local_agent.config
        assert config["name"] == "test-agent"
        assert config["model"] == "gpt-4"
        assert config["instructions"] == "You are a helpful assistant."
        assert config["max_steps"] == 5


class TestLocalAgentReload:
    """Tests for reloading existing LocalAgent."""

    def test_reload_existing(self, real_agent, temp_storage):
        """Test loading an existing LocalAgent by name."""
        LocalAgent.from_agent(real_agent, "test", temp_storage)

        # Reload
        local_agent = LocalAgent("test", temp_storage)

        assert local_agent.name == "test"
        assert local_agent.config["name"] == "test-agent"

    def test_reload_specific_version(self, real_agent, temp_storage):
        """Test loading specific version."""
        LocalAgent.from_agent(
            real_agent,
            "test",
            temp_storage,
            version="1.0.0",
        )

        local_agent = LocalAgent("test", temp_storage, version="1.0.0")

        assert local_agent.version == "1.0.0"

    def test_reload_nonexistent_raises(self, temp_storage):
        """Test that loading nonexistent agent raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            LocalAgent("nonexistent", temp_storage)


class TestLocalAgentLoad:
    """Tests for LocalAgent.load() method."""

    def test_load_returns_agent(self, real_agent, temp_storage):
        """Test that load() returns an Agent instance."""
        local_agent = LocalAgent.from_agent(real_agent, "test", temp_storage)

        loaded_agent = local_agent.load()

        assert isinstance(loaded_agent, Agent)
        assert loaded_agent.name == "test-agent"
        assert loaded_agent.max_steps == 5

    def test_load_with_overrides(self, real_agent, temp_storage):
        """Test load() with override arguments."""
        local_agent = LocalAgent.from_agent(real_agent, "test", temp_storage)

        loaded_agent = local_agent.load(max_steps=20)

        assert loaded_agent.max_steps == 20


class TestLocalAgentMethods:
    """Tests for LocalAgent helper methods."""

    def test_publish_not_implemented(self, real_agent, temp_storage):
        """Test that publish() raises NotImplementedError."""
        local_agent = LocalAgent.from_agent(real_agent, "test", temp_storage)

        with pytest.raises(NotImplementedError):
            local_agent.publish()

    def test_repr(self, real_agent, temp_storage):
        """Test __repr__ method."""
        local_agent = LocalAgent.from_agent(real_agent, "test", temp_storage)

        repr_str = repr(local_agent)
        assert "LocalAgent" in repr_str
        assert "test" in repr_str
        assert "0.1.0" in repr_str


# ==============================================================================
# AgentPackage Tests (Mock-based since no real packages)
# ==============================================================================


class TestAgentPackage:
    """Tests for AgentPackage class."""

    def test_package_properties(self):
        """Test that package exposes manifest properties."""
        with patch.object(AgentPackage, "_find_entry_point") as mock_ep:
            mock_ep.return_value = MagicMock()

            with patch.object(
                AgentPackage,
                "manifest",
                new_callable=lambda: property(
                    lambda self: AgentManifest(
                        entrypoint="main:agent",
                        toolsets=["tools1"],
                        models=["gpt-4"],
                        datasets=["data1"],
                    )
                ),
            ):
                # Can't fully test without real entry points
                pass


# ==============================================================================
# Integration with dn.load()
# ==============================================================================


class TestUnifiedLoadAgent:
    """Tests for unified dn.load() with agents."""

    def test_load_agent_scheme_calls_loader(self):
        """Test that agent:// scheme uses AgentPackage."""
        from dreadnode.core.load import load

        with patch("dreadnode.agents.loader.AgentPackage") as MockPackage:
            mock_pkg = MagicMock()
            mock_agent = MagicMock(spec=Agent)
            mock_pkg.load.return_value = mock_agent
            MockPackage.return_value = mock_pkg

            result = load("agent://my-agent")

            MockPackage.assert_called_once_with("my-agent")
            mock_pkg.load.assert_called_once()
            assert result == mock_agent


# ==============================================================================
# Edge Cases
# ==============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_agent_without_model(self, temp_storage):
        """Test storing agent without model."""
        agent = Agent(
            name="no-model-agent",
            description="Agent without model",
            instructions="Test",
        )

        local_agent = LocalAgent.from_agent(agent, "test", temp_storage)

        assert local_agent.config["model"] is None

    def test_manifest_fields(self, real_agent, temp_storage):
        """Test that manifest has expected fields."""
        local_agent = LocalAgent.from_agent(real_agent, "test", temp_storage)

        manifest = local_agent.manifest
        assert manifest.entrypoint == "config:agent"
        assert isinstance(manifest.toolsets, list)
        assert isinstance(manifest.models, list)
        assert isinstance(manifest.datasets, list)
