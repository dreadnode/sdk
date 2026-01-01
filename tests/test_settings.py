"""Tests for dreadnode.core.settings module."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from dreadnode.core.settings import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MAX_WORKERS,
    DEFAULT_PLATFORM_BASE_DOMAIN,
    SETTINGS_FILE,
    DreadnodeSettings,
    create_settings,
    get_settings_file_path,
    load_yaml_settings,
    settings,
)


# Environment variables that might affect settings tests
DREADNODE_ENV_VARS = [
    "DREADNODE_SERVER",
    "DREADNODE_SERVER_URL",
    "DREADNODE_TOKEN",
    "DREADNODE_API_TOKEN",
    "DREADNODE_API_KEY",
    "DREADNODE_DEBUG",
    "DREADNODE_ORGANIZATION",
    "DREADNODE_WORKSPACE",
    "DREADNODE_PROJECT",
    "DREADNODE_MAX_WORKERS",
]


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all DREADNODE_ environment variables for clean tests."""
    for var in DREADNODE_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class TestLoadYamlSettings:
    """Tests for load_yaml_settings function."""

    def test_returns_empty_dict_when_file_not_exists(self) -> None:
        """Should return empty dict when settings file doesn't exist."""
        result = load_yaml_settings(Path("/nonexistent/path/settings.yaml"))
        assert result == {}

    def test_loads_valid_yaml_file(self, tmp_path: Path) -> None:
        """Should load settings from a valid YAML file."""
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(
            yaml.dump(
                {
                    "debug": True,
                    "organization": "test-org",
                    "max_workers": 20,
                }
            )
        )

        result = load_yaml_settings(settings_file)

        assert result == {
            "debug": True,
            "organization": "test-org",
            "max_workers": 20,
        }

    def test_returns_empty_dict_for_empty_yaml_file(self, tmp_path: Path) -> None:
        """Should return empty dict for empty YAML file."""
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("")

        result = load_yaml_settings(settings_file)

        assert result == {}

    def test_returns_empty_dict_for_non_dict_yaml(self, tmp_path: Path) -> None:
        """Should return empty dict when YAML content is not a dict."""
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text("- item1\n- item2")  # YAML list, not dict

        result = load_yaml_settings(settings_file)

        assert result == {}

    def test_uses_default_path_when_none(self) -> None:
        """Should use default path when settings_path is None."""
        # This should not raise, even if default file doesn't exist
        result = load_yaml_settings(None)
        assert isinstance(result, dict)


class TestGetSettingsFilePath:
    """Tests for get_settings_file_path function."""

    def test_returns_default_path(self) -> None:
        """Should return path based on default cache dir."""
        result = get_settings_file_path()
        assert result == DEFAULT_CACHE_DIR / SETTINGS_FILE

    def test_uses_custom_cache_dir(self, tmp_path: Path) -> None:
        """Should use custom cache dir when provided."""
        result = get_settings_file_path(tmp_path)
        assert result == tmp_path / SETTINGS_FILE


class TestDreadnodeSettings:
    """Tests for DreadnodeSettings class."""

    def test_default_values(self, clean_env: None) -> None:
        """Should have sensible default values."""
        s = DreadnodeSettings()

        assert s.cache_dir == DEFAULT_CACHE_DIR
        assert s.max_workers == DEFAULT_MAX_WORKERS
        assert s.platform_base_domain == DEFAULT_PLATFORM_BASE_DOMAIN
        assert s.debug is False
        assert s.server is None
        assert s.token is None

    def test_server_url_property_with_default(self, clean_env: None) -> None:
        """Should compute server_url from platform_base_domain."""
        s = DreadnodeSettings()
        assert s.server_url == f"https://platform.{DEFAULT_PLATFORM_BASE_DOMAIN}"

    def test_server_url_property_with_custom_server(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use custom server when provided via env var."""
        # server field uses validation_alias, so must be set via env var
        monkeypatch.setenv("DREADNODE_SERVER", "https://custom.example.com")
        s = DreadnodeSettings()
        assert s.server_url == "https://custom.example.com"

    def test_settings_file_path_property(self) -> None:
        """Should return correct settings file path."""
        s = DreadnodeSettings()
        assert s.settings_file_path == s.cache_dir / SETTINGS_FILE

    def test_has_env_credentials_false(self, clean_env: None) -> None:
        """Should return False when no credentials set."""
        s = DreadnodeSettings()
        assert s.has_env_credentials is False

    def test_has_env_credentials_with_server(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return True when server is set via env var."""
        # server field uses validation_alias, so must be set via env var
        monkeypatch.setenv("DREADNODE_SERVER", "https://example.com")
        s = DreadnodeSettings()
        assert s.has_env_credentials is True

    def test_has_env_credentials_with_token(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should return True when token is set via env var."""
        # token field uses validation_alias, so must be set via env var
        monkeypatch.setenv("DREADNODE_API_TOKEN", "test-token")
        s = DreadnodeSettings()
        assert s.has_env_credentials is True

    def test_loads_from_environment_variables(
        self, clean_env: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should load settings from environment variables."""
        monkeypatch.setenv("DREADNODE_DEBUG", "true")
        monkeypatch.setenv("DREADNODE_ORGANIZATION", "env-org")
        monkeypatch.setenv("DREADNODE_MAX_WORKERS", "50")

        s = DreadnodeSettings()

        assert s.debug is True
        assert s.organization == "env-org"
        assert s.max_workers == 50


class TestCreateSettings:
    """Tests for create_settings function."""

    def test_creates_settings_without_yaml(self, clean_env: None) -> None:
        """Should create settings with defaults when no YAML exists."""
        s = create_settings(Path("/nonexistent/settings.yaml"))
        assert s.debug is False
        assert s.max_workers == DEFAULT_MAX_WORKERS

    def test_creates_settings_from_yaml(self, clean_env: None, tmp_path: Path) -> None:
        """Should create settings with values from YAML file."""
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(
            yaml.dump(
                {
                    "debug": True,
                    "organization": "yaml-org",
                    "max_workers": 25,
                }
            )
        )

        s = create_settings(settings_file)

        assert s.debug is True
        assert s.organization == "yaml-org"
        assert s.max_workers == 25

    def test_env_vars_override_yaml(
        self, clean_env: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Environment variables should override YAML values."""
        settings_file = tmp_path / "settings.yaml"
        settings_file.write_text(
            yaml.dump(
                {
                    "debug": False,
                    "organization": "yaml-org",
                    "max_workers": 25,
                }
            )
        )

        # Set env var to override YAML
        monkeypatch.setenv("DREADNODE_DEBUG", "true")
        monkeypatch.setenv("DREADNODE_MAX_WORKERS", "100")

        s = create_settings(settings_file)

        # These should be overridden by env vars
        assert s.debug is True
        assert s.max_workers == 100
        # This should come from YAML
        assert s.organization == "yaml-org"


class TestGlobalSettings:
    """Tests for the global settings singleton."""

    def test_global_settings_exists(self) -> None:
        """Should have a global settings instance."""
        assert settings is not None
        assert isinstance(settings, DreadnodeSettings)

    def test_global_settings_has_expected_properties(self) -> None:
        """Global settings should have all expected properties."""
        assert hasattr(settings, "cache_dir")
        assert hasattr(settings, "server_url")
        assert hasattr(settings, "settings_file_path")
        assert hasattr(settings, "user_agent")
