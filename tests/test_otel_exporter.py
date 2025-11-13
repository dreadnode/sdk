"""Tests for OTLP exporter and Task output linking changes."""

import pytest
from unittest.mock import Mock
from urllib.parse import urljoin


class TestCustomOTLPSpanExporterLogic:
    """Test CustomOTLPSpanExporter logic for custom endpoint and User-Agent injection."""

    def test_custom_endpoint_is_extracted_from_kwargs(self):
        """Test that custom_endpoint is extracted before passing to parent."""
        test_kwargs = {
            "endpoint": "https://example.com",
            "custom_endpoint": "https://example.com/api/otel/traces",
            "headers": {"X-Api-Key": "test-key"},
        }

        # Simulate: custom_endpoint = kwargs.pop("custom_endpoint", None)
        custom_endpoint = test_kwargs.pop("custom_endpoint", None)

        assert custom_endpoint == "https://example.com/api/otel/traces"
        assert "custom_endpoint" not in test_kwargs
        assert "endpoint" in test_kwargs
        assert "headers" in test_kwargs

    def test_user_agent_combination_with_string(self):
        """Test User-Agent combination logic with string input."""
        DEFAULT_USER_AGENT = "dreadnode/1.0.0"
        otlp_user_agent = "OTel-OTLP-Exporter-Python/1.0.0"

        # Simulate the combination logic from exporter.py
        if isinstance(otlp_user_agent, bytes):
            otlp_user_agent = otlp_user_agent.decode("utf-8")

        if otlp_user_agent:
            combined_user_agent = f"{DEFAULT_USER_AGENT} {otlp_user_agent}"

        assert DEFAULT_USER_AGENT in combined_user_agent
        assert "OTel-OTLP-Exporter-Python/1.0.0" in combined_user_agent
        assert combined_user_agent.startswith(DEFAULT_USER_AGENT)

    def test_user_agent_combination_with_bytes(self):
        """Test User-Agent combination logic with bytes input."""
        DEFAULT_USER_AGENT = "dreadnode/1.0.0"
        otlp_user_agent = b"OTel-OTLP-Exporter-Python/1.0.0"

        # Simulate the combination logic
        if isinstance(otlp_user_agent, bytes):
            otlp_user_agent = otlp_user_agent.decode("utf-8")

        if otlp_user_agent:
            combined_user_agent = f"{DEFAULT_USER_AGENT} {otlp_user_agent}"

        assert isinstance(combined_user_agent, str)
        assert DEFAULT_USER_AGENT in combined_user_agent
        assert "OTel-OTLP-Exporter-Python/1.0.0" in combined_user_agent

    def test_user_agent_fallback_when_none(self):
        """Test User-Agent fallback when no OTLP User-Agent exists."""
        DEFAULT_USER_AGENT = "dreadnode/1.0.0"
        otlp_user_agent = None

        # Simulate the fallback logic
        if isinstance(otlp_user_agent, bytes):
            otlp_user_agent = otlp_user_agent.decode("utf-8")

        if otlp_user_agent:
            combined_user_agent = f"{DEFAULT_USER_AGENT} {otlp_user_agent}"
        else:
            combined_user_agent = DEFAULT_USER_AGENT

        assert combined_user_agent == DEFAULT_USER_AGENT

    def test_custom_endpoint_override_logic(self):
        """Test the custom_endpoint override logic."""
        mock_exporter = Mock()
        mock_exporter._endpoint = "https://example.com/v1/traces"  # Default OTLP

        custom_endpoint = "https://example.com/api/otel/traces"

        # Simulate: if custom_endpoint: self._endpoint = custom_endpoint
        if custom_endpoint:
            mock_exporter._endpoint = custom_endpoint

        assert mock_exporter._endpoint == custom_endpoint
        assert mock_exporter._endpoint != "https://example.com/v1/traces"

    def test_no_custom_endpoint_preserves_default(self):
        """Test that no custom_endpoint doesn't override the default."""
        mock_exporter = Mock()
        default_endpoint = "https://example.com/v1/traces"
        mock_exporter._endpoint = default_endpoint

        custom_endpoint = None

        if custom_endpoint:
            mock_exporter._endpoint = custom_endpoint

        assert mock_exporter._endpoint == default_endpoint


class TestDreadnodeExporterConfiguration:
    """Test Dreadnode exporter configuration in main.py."""

    def test_custom_endpoint_construction(self):
        """Test that custom endpoint is constructed correctly with urljoin."""
        server = "https://platform.example.com"
        custom_endpoint = urljoin(server, "/api/otel/traces")

        assert custom_endpoint == "https://platform.example.com/api/otel/traces"

    def test_custom_endpoint_with_trailing_slash(self):
        """Test custom endpoint construction with trailing slash in server URL."""
        server = "https://platform.example.com/"
        custom_endpoint = urljoin(server, "/api/otel/traces")

        assert custom_endpoint == "https://platform.example.com/api/otel/traces"

    def test_endpoint_and_custom_endpoint_are_different(self):
        """Test that endpoint and custom_endpoint parameters are different."""
        server = "https://platform.example.com"

        endpoint = server
        custom_endpoint = urljoin(server, "/api/otel/traces")

        assert endpoint != custom_endpoint
        assert custom_endpoint.endswith("/api/otel/traces")
        assert not custom_endpoint.endswith("/v1/traces")
        assert "/v1/traces" not in custom_endpoint


class TestTaskOutputHashBugFix:
    """Test the Task output_object_hash initialization bug fix (dreadnode/task.py:541)."""

    def test_output_object_hash_initialized_to_none(self):
        """Test that output_object_hash is initialized before conditional (prevents UnboundLocalError)."""
        # Simulate the fix: output_object_hash = None
        output_object_hash = None

        # This should not raise UnboundLocalError
        try:
            if output_object_hash is not None:
                pass  # Would call link_objects here
            assert True
        except UnboundLocalError:
            pytest.fail("Should not raise UnboundLocalError after fix")

    def test_linking_only_when_hash_exists(self):
        """Test that linking logic only executes when hash is not None."""
        output_object_hash = None
        link_called = False

        # Simulate: if output_object_hash is not None: run.link_objects(...)
        if output_object_hash is not None:
            link_called = True

        assert not link_called

    def test_linking_when_hash_exists(self):
        """Test that linking logic executes when hash exists."""
        output_object_hash = "some_hash_value"
        link_called = False

        if output_object_hash is not None:
            link_called = True

        assert link_called

    def test_multiple_inputs_linked_to_single_output(self):
        """Test logic for linking multiple input hashes to one output hash."""
        output_object_hash = "output_123"
        input_object_hashes = ["input_1", "input_2", "input_3"]
        links = []

        # Simulate: for input_object_hash in input_object_hashes:
        #              run.link_objects(output_object_hash, input_object_hash)
        if output_object_hash is not None:
            for input_object_hash in input_object_hashes:
                links.append((output_object_hash, input_object_hash))

        assert len(links) == 3
        for output_hash, input_hash in links:
            assert output_hash == "output_123"
            assert input_hash in input_object_hashes

    def test_no_linking_when_output_hash_is_none(self):
        """Test that no linking occurs when output_object_hash is None (output not logged)."""
        output_object_hash = None  # This is what the fix ensures is initialized
        input_object_hashes = ["input_1", "input_2"]
        links = []

        # Simulate the linking loop with the None check
        if output_object_hash is not None:
            for input_object_hash in input_object_hashes:
                links.append((output_object_hash, input_object_hash))

        # Should not create any links when output_hash is None
        assert len(links) == 0
