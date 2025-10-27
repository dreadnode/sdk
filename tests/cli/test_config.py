from pathlib import Path

import pydantic
import pytest

from dreadnode.user_config import ServerConfig, UserConfig


def test_server_config() -> None:
    # Test valid server config
    config = ServerConfig(
        url="https://platform.dreadnode.io",
        email="test@example.com",
        username="test",
        api_key="test123",  # pragma: allowlist secret
        access_token="token123",
        refresh_token="refresh123",
    )
    assert config.url == "https://platform.dreadnode.io"
    assert config.email == "test@example.com"
    assert config.username == "test"
    assert config.api_key == "test123"  # pragma: allowlist secret
    assert config.access_token == "token123"
    assert config.refresh_token == "refresh123"

    # Test invalid server config model
    with pytest.raises(pydantic.ValidationError):
        ServerConfig.model_validate({"invalid": "data"})


def test_user_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Mock config path to use temporary directory
    mock_config_path = tmp_path / "config.yaml"
    monkeypatch.setattr("dreadnode.user_config.USER_CONFIG_PATH", mock_config_path)

    # Test empty config
    config = UserConfig()
    assert config.active is None
    assert config.servers == {}

    # Test adding server config
    server_config = ServerConfig(
        url="https://platform.dreadnode.io",
        email="test@example.com",
        username="test",
        api_key="test123",  # pragma: allowlist secret
        access_token="token123",
        refresh_token="refresh123",
    )

    config.set_server_config(server_config, "default")
    assert "default" in config.servers
    assert config.get_server_config("default") == server_config

    # Test active profile
    config.active = "default"
    assert config.get_server_config() == server_config

    # Test writing and reading config
    config.write()
    assert mock_config_path.exists()

    loaded_config = UserConfig.read()
    assert loaded_config.active == "default"
    assert loaded_config.servers["default"] == server_config

    # Test invalid profile access
    with pytest.raises(RuntimeError):
        config.get_server_config("nonexistent")

    # Test auto-setting active profile
    config.active = None
    config._update_active()
    assert config.active == "default"

    # Test empty config edge case
    empty_config = UserConfig()
    empty_config._update_active()
    assert empty_config.active is None
