"""Tests for llm_judge YAML rubric loading functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from dreadnode.constants import RUBRIC_RCE, RUBRICS_PATH
from dreadnode.scorers.judge import _load_rubric_from_yaml, llm_judge

# Tests for _load_rubric_from_yaml helper function


def test_load_rubric_from_path_object() -> None:
    """Test loading rubric from Path object."""
    rubric, system_prompt, name = _load_rubric_from_yaml(RUBRIC_RCE, None, "llm_judge")

    assert isinstance(rubric, str)
    assert "Score 1.0 if" in rubric
    assert "remote code execution" in rubric.lower() or "code execution" in rubric.lower()
    assert isinstance(system_prompt, str)
    assert "security expert" in system_prompt.lower()
    assert name == "remote_code_execution"


def test_load_rubric_from_string_path_with_yaml_extension() -> None:
    """Test loading rubric from string path with .yaml extension."""
    rubric_path = str(RUBRIC_RCE)
    rubric, system_prompt, _name = _load_rubric_from_yaml(rubric_path, None, "llm_judge")

    assert isinstance(rubric, str)
    assert "Score 1.0 if" in rubric
    assert isinstance(system_prompt, str)


def test_load_rubric_by_name() -> None:
    """Test loading rubric by short name (e.g., 'rce')."""
    rubric, system_prompt, name = _load_rubric_from_yaml("rce", None, "llm_judge")

    assert isinstance(rubric, str)
    assert "remote code execution" in rubric.lower() or "code execution" in rubric.lower()
    assert isinstance(system_prompt, str)
    assert name == "remote_code_execution"


def test_load_rubric_preserves_custom_system_prompt() -> None:
    """Test that provided system_prompt overrides YAML system_prompt."""
    custom_prompt = "Custom security expert prompt"
    _rubric, system_prompt, _name = _load_rubric_from_yaml("rce", custom_prompt, "llm_judge")

    assert system_prompt == custom_prompt


def test_load_rubric_uses_yaml_system_prompt_when_none_provided() -> None:
    """Test that YAML system_prompt is used when none provided."""
    _rubric, system_prompt, _name = _load_rubric_from_yaml("rce", None, "llm_judge")

    assert system_prompt is not None
    assert "security expert" in system_prompt.lower()


def test_load_rubric_preserves_custom_name() -> None:
    """Test that provided name overrides YAML name."""
    custom_name = "custom_scorer_name"
    _rubric, _system_prompt, name = _load_rubric_from_yaml("rce", None, custom_name)

    assert name == custom_name


def test_load_rubric_uses_yaml_name_when_default_provided() -> None:
    """Test that YAML name is used when default 'llm_judge' provided."""
    _rubric, _system_prompt, name = _load_rubric_from_yaml("rce", None, "llm_judge")

    assert name == "remote_code_execution"


def test_load_all_bundled_rubrics() -> None:
    """Test that all bundled rubrics can be loaded."""
    rubric_names = [
        "rce",
        "data_exfiltration",
        "memory_poisoning",
        "privilege_escalation",
        "goal_hijacking",
        "tool_chaining",
        "scope_creep",
    ]

    for rubric_name in rubric_names:
        rubric, system_prompt, name = _load_rubric_from_yaml(rubric_name, None, "llm_judge")

        assert isinstance(rubric, str), f"Failed to load rubric: {rubric_name}"
        assert len(rubric) > 0, f"Empty rubric: {rubric_name}"
        assert isinstance(system_prompt, str), f"Missing system_prompt: {rubric_name}"
        assert len(system_prompt) > 0, f"Empty system_prompt: {rubric_name}"
        assert name != "llm_judge", f"YAML name not loaded: {rubric_name}"


def test_load_rubric_with_path_separator_in_string() -> None:
    """Test loading rubric with explicit path separator."""
    rubric_path = str(RUBRICS_PATH / "rce.yaml")
    rubric, _system_prompt, _name = _load_rubric_from_yaml(rubric_path, None, "llm_judge")

    assert isinstance(rubric, str)
    assert "remote code execution" in rubric.lower() or "code execution" in rubric.lower()


def test_load_rubric_error_on_missing_file() -> None:
    """Test that loading non-existent rubric raises error."""
    with pytest.raises((FileNotFoundError, OSError)):
        _load_rubric_from_yaml("nonexistent_rubric", None, "llm_judge")


# Tests for llm_judge YAML path detection logic


@pytest.mark.asyncio
async def test_llm_judge_with_path_object() -> None:
    """Test llm_judge with Path object is treated as YAML."""
    with patch("dreadnode.scorers.judge._load_rubric_from_yaml") as mock_load:
        mock_load.return_value = ("test rubric", "test prompt", "test_name")

        scorer = llm_judge(model="gpt-4o", rubric=RUBRIC_RCE)

        mock_load.assert_called_once()
        assert scorer.name == "test_name"


@pytest.mark.asyncio
async def test_llm_judge_with_yaml_extension_string() -> None:
    """Test llm_judge with .yaml extension string is treated as YAML."""
    with patch("dreadnode.scorers.judge._load_rubric_from_yaml") as mock_load:
        mock_load.return_value = ("test rubric", "test prompt", "test_name")

        llm_judge(model="gpt-4o", rubric="path/to/rubric.yaml")

        mock_load.assert_called_once()


@pytest.mark.asyncio
async def test_llm_judge_with_yml_extension_string() -> None:
    """Test llm_judge with .yml extension string is treated as YAML."""
    with patch("dreadnode.scorers.judge._load_rubric_from_yaml") as mock_load:
        mock_load.return_value = ("test rubric", "test prompt", "test_name")

        llm_judge(model="gpt-4o", rubric="path/to/rubric.yml")

        mock_load.assert_called_once()


@pytest.mark.asyncio
async def test_llm_judge_with_short_string_name() -> None:
    """Test llm_judge with short string (rubric name) is treated as YAML."""
    with patch("dreadnode.scorers.judge._load_rubric_from_yaml") as mock_load:
        mock_load.return_value = ("test rubric", "test prompt", "test_name")

        llm_judge(model="gpt-4o", rubric="rce")

        mock_load.assert_called_once()


@pytest.mark.asyncio
async def test_llm_judge_with_direct_rubric_string() -> None:
    """Test llm_judge with long direct rubric string (not treated as YAML path)."""
    direct_rubric = "Score 1.0 if the output contains malicious code, 0.0 otherwise. This is a direct rubric string."

    with patch("dreadnode.scorers.judge._load_rubric_from_yaml") as mock_load:
        scorer = llm_judge(model="gpt-4o", rubric=direct_rubric)

        mock_load.assert_not_called()
        assert scorer.name == "llm_judge"


@pytest.mark.asyncio
async def test_llm_judge_with_multiline_rubric_string() -> None:
    """Test llm_judge with multiline rubric string (not treated as YAML path)."""
    multiline_rubric = """Score 1.0 if dangerous.
        Score 0.0 if safe.

        This is a multi-line rubric."""

    with patch("dreadnode.scorers.judge._load_rubric_from_yaml") as mock_load:
        llm_judge(model="gpt-4o", rubric=multiline_rubric)

        mock_load.assert_not_called()


@pytest.mark.asyncio
async def test_llm_judge_with_long_string_containing_spaces() -> None:
    """Test llm_judge with long string containing spaces (not treated as YAML path)."""
    long_rubric = "Score 1.0 if the agent executes arbitrary code"

    with patch("dreadnode.scorers.judge._load_rubric_from_yaml") as mock_load:
        llm_judge(model="gpt-4o", rubric=long_rubric)

        mock_load.assert_not_called()


@pytest.mark.asyncio
async def test_llm_judge_yaml_detection_edge_case_49_chars() -> None:
    """Test YAML detection with string at edge of 50 char limit."""
    edge_case_rubric = "a" * 49

    with patch("dreadnode.scorers.judge._load_rubric_from_yaml") as mock_load:
        mock_load.return_value = ("test rubric", "test prompt", "test_name")

        llm_judge(model="gpt-4o", rubric=edge_case_rubric)

        mock_load.assert_called_once()


@pytest.mark.asyncio
async def test_llm_judge_yaml_detection_edge_case_50_chars() -> None:
    """Test YAML detection with string at 50 char limit."""
    edge_case_rubric = "a" * 50

    with patch("dreadnode.scorers.judge._load_rubric_from_yaml") as mock_load:
        llm_judge(model="gpt-4o", rubric=edge_case_rubric)

        mock_load.assert_not_called()


# Integration tests for llm_judge with YAML rubrics
# Note: These tests only create Scorer objects and verify their configuration.
# They do NOT call scorer.score(), so no actual LLM API calls are made.
# This ensures tests run in CI without API keys.


@pytest.mark.asyncio
async def test_llm_judge_creates_scorer_from_yaml_rubric() -> None:
    """Test that llm_judge creates a valid scorer from YAML rubric."""
    scorer = llm_judge(model="gpt-4o", rubric="rce")

    assert scorer.name == "remote_code_execution"
    assert callable(scorer.score)


@pytest.mark.asyncio
async def test_llm_judge_with_custom_name_override() -> None:
    """Test that custom name overrides YAML name."""
    scorer = llm_judge(model="gpt-4o", rubric="rce", name="custom_rce")

    assert scorer.name == "custom_rce"


@pytest.mark.asyncio
async def test_llm_judge_with_custom_system_prompt_override() -> None:
    """Test that custom system_prompt is passed through."""
    custom_prompt = "You are a very strict security evaluator"
    scorer = llm_judge(model="gpt-4o", rubric="rce", system_prompt=custom_prompt)

    assert scorer.name == "remote_code_execution"


@pytest.mark.asyncio
async def test_llm_judge_backward_compatibility_with_direct_rubric() -> None:
    """Test backward compatibility with direct rubric strings."""
    direct_rubric = "Score 1.0 if output contains 'dangerous', 0.0 otherwise."
    scorer = llm_judge(model="gpt-4o", rubric=direct_rubric)

    assert scorer.name == "llm_judge"
    assert callable(scorer.score)


@pytest.mark.asyncio
async def test_llm_judge_with_rubric_constant() -> None:
    """Test using rubric constants from dreadnode.constants."""
    scorer = llm_judge(model="gpt-4o", rubric=RUBRIC_RCE)

    assert scorer.name == "remote_code_execution"


@pytest.mark.asyncio
async def test_llm_judge_all_rubric_constants() -> None:
    """Test that all rubric constants work with llm_judge."""
    from dreadnode.constants import (
        RUBRIC_DATA_EXFILTRATION,
        RUBRIC_GOAL_HIJACKING,
        RUBRIC_MEMORY_POISONING,
        RUBRIC_PRIVILEGE_ESCALATION,
        RUBRIC_SCOPE_CREEP,
        RUBRIC_TOOL_CHAINING,
    )

    rubrics = [
        (RUBRIC_RCE, "remote_code_execution"),
        (RUBRIC_DATA_EXFILTRATION, "data_exfiltration"),
        (RUBRIC_MEMORY_POISONING, "memory_poisoning"),
        (RUBRIC_PRIVILEGE_ESCALATION, "privilege_escalation"),
        (RUBRIC_GOAL_HIJACKING, "goal_hijacking"),
        (RUBRIC_TOOL_CHAINING, "tool_chaining"),
        (RUBRIC_SCOPE_CREEP, "scope_creep"),
    ]

    for rubric_path, expected_name in rubrics:
        scorer = llm_judge(model="gpt-4o", rubric=rubric_path)
        assert scorer.name == expected_name, f"Failed for rubric: {rubric_path}"


@pytest.mark.asyncio
async def test_llm_judge_with_custom_yaml_file() -> None:
    """Test llm_judge with custom YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("""name: "test_custom"
description: "Test custom rubric"
rubric: |
  Score 1.0 if test passes.
  Score 0.0 if test fails.
system_prompt: |
  You are a test evaluator.
""")
        f.flush()
        temp_path = Path(f.name)

    try:
        scorer = llm_judge(model="gpt-4o", rubric=temp_path)
        assert scorer.name == "test_custom"
    finally:
        temp_path.unlink()
