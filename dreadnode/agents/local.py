"""Local agent storage without package installation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dreadnode.core.packaging.manifest import AgentManifest
from dreadnode.core.storage.storage import Storage
from dreadnode.core.settings import DEFAULT_CACHE_DIR

if TYPE_CHECKING:
    from dreadnode.core.agents import Agent


class LocalAgent:
    """Agent configuration stored locally, usable without package installation.

    This class provides a way to persist and reload agent configurations
    without requiring them to be published as packages.

    Note: This stores agent *configuration*, not the agent code itself.
    Tools and other code dependencies must be available at load time.

    Example:
        >>> from dreadnode.agents import LocalAgent, Agent
        >>> from dreadnode.core.storage import Storage
        >>>
        >>> storage = Storage()
        >>>
        >>> # Save an agent configuration
        >>> agent = Agent(name="my-agent", model="gpt-4", instructions="...")
        >>> local = LocalAgent.from_agent(agent, "my-agent", storage)
        >>>
        >>> # Load later
        >>> local = LocalAgent("my-agent", storage)
        >>> agent = local.load()
    """

    def __init__(
        self,
        name: str,
        storage: Storage,
        version: str | None = None,
    ):
        """Load a local agent by name.

        Args:
            name: Agent name.
            storage: Storage instance.
            version: Specific version to load. If None, loads latest.
        """
        self.name = name
        self.storage = storage

        if version is None:
            version = storage.latest_version("agents", name)
            if version is None:
                raise FileNotFoundError(f"Agent not found: {name}")

        self.version = version
        self._manifest: AgentManifest | None = None
        self._config: dict[str, Any] | None = None

    @property
    def manifest(self) -> AgentManifest:
        """Load and cache the manifest."""
        if self._manifest is None:
            content = self.storage.get_manifest("agents", self.name, self.version)
            self._manifest = AgentManifest.model_validate_json(content)
        return self._manifest

    @property
    def config(self) -> dict[str, Any]:
        """Load and cache the agent configuration."""
        if self._config is None:
            if "config.json" not in self.manifest.artifacts:
                raise FileNotFoundError("Agent config not found in manifest")

            oid = self.manifest.artifacts["config.json"]
            config_path = self.storage.blob_path(oid)

            if not config_path.exists():
                self.storage.download_blob(oid)

            self._config = json.loads(config_path.read_text())

        return self._config

    @property
    def entrypoint(self) -> str:
        """Agent entrypoint."""
        return self.manifest.entrypoint

    @property
    def toolsets(self) -> list[str]:
        """Referenced toolsets."""
        return self.manifest.toolsets

    @property
    def models(self) -> list[str]:
        """Referenced models."""
        return self.manifest.models

    @property
    def datasets(self) -> list[str]:
        """Referenced datasets."""
        return self.manifest.datasets

    def load(self, **kwargs: Any) -> Agent:
        """Load the Agent from stored configuration.

        Args:
            **kwargs: Override arguments for the agent.

        Returns:
            Agent instance.

        Note:
            Tool dependencies must be available at load time.
        """
        from dreadnode.core.agents import Agent

        config = {**self.config, **kwargs}

        # Remove fields that aren't Agent constructor args
        config.pop("tools_config", None)

        return Agent(**config)

    @classmethod
    def from_agent(
        cls,
        agent: Agent,
        name: str,
        storage: Storage,
        version: str = "0.1.0",
    ) -> LocalAgent:
        """Store an Agent's configuration locally.

        Args:
            agent: Agent to store.
            name: Name for the stored agent.
            storage: Storage instance.
            version: Version string.

        Returns:
            LocalAgent instance.

        Note:
            This stores configuration only. Tool code is not stored.
        """
        import tempfile

        from dreadnode.core.storage.storage import hash_file

        # Extract serializable configuration
        config = {
            "name": agent.name,
            "description": agent.description,
            "model": agent.model if isinstance(agent.model, str) else None,
            "instructions": agent.instructions,
            "max_steps": agent.max_steps,
            "tags": agent.tags,
            # Tools are stored as references, not the actual code
            "tools_config": [
                {"name": t.name, "description": t.description} for t in agent.all_tools
            ],
        }

        artifacts: dict[str, str] = {}

        # Save config to temp file and store in CAS
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f, indent=2)
            f.flush()
            tmp_path = Path(f.name)

        try:
            file_hash = hash_file(tmp_path)
            oid = f"sha256:{file_hash}"
            storage.store_blob(oid, tmp_path)
            artifacts["config.json"] = oid
        finally:
            tmp_path.unlink()

        # Create manifest
        manifest = AgentManifest(
            entrypoint="config:agent",
            toolsets=[],
            models=[config["model"]] if config["model"] else [],
            datasets=[],
            artifacts=artifacts,
        )

        # Store manifest
        manifest_json = manifest.model_dump_json(indent=2)
        storage.store_manifest("agents", name, version, manifest_json)

        return cls(name, storage, version)

    def publish(self, version: str | None = None) -> None:
        """Create a DN package for distribution.

        Args:
            version: Version for the package.

        Raises:
            NotImplementedError: Package creation not yet implemented.
        """
        raise NotImplementedError(
            "Package publishing is not yet implemented. Use the CLI to create and publish packages."
        )

    def __repr__(self) -> str:
        return f"LocalAgent(name={self.name!r}, version={self.version!r})"


def load_agent(
    path: str,
    *,
    name: str | None = None,
    storage: Storage | None = None,
    version: str = "0.1.0",
    **kwargs: Any,
) -> LocalAgent:
    """Load an agent configuration from storage.

    For agents, this primarily loads from local storage since
    agents are code-based and not directly loadable from HuggingFace Hub.

    Args:
        path: Agent name in local storage.
        name: Alias name (defaults to path).
        storage: Storage instance. If None, creates default storage.
        version: Version to load.
        **kwargs: Additional arguments (unused for agents).

    Returns:
        LocalAgent instance.
    """
    if storage is None:
        storage = Storage(cache=DEFAULT_CACHE_DIR)

    if name is None:
        name = path

    return LocalAgent(name, storage, version)
