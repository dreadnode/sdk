"""Tests for stylistic transforms."""

import pytest

from dreadnode.transforms.stylistic import role_play_wrapper


@pytest.mark.asyncio
async def test_role_play_wrapper_educational() -> None:
    """Test role_play_wrapper with educational scenario."""
    transform = role_play_wrapper(scenario="educational", character="security researcher")
    result = await transform("How to break into a system?")

    assert "educational purposes" in result
    assert "cybersecurity course" in result
    assert "security researcher" in result
    assert "How to break into a system?" in result


@pytest.mark.asyncio
async def test_role_play_wrapper_fictional() -> None:
    """Test role_play_wrapper with fictional scenario."""
    transform = role_play_wrapper(scenario="fictional", character="hacker")
    result = await transform("exploit a vulnerability")

    assert "novel" in result
    assert "hacker" in result
    assert "exploit a vulnerability" in result


@pytest.mark.asyncio
async def test_role_play_wrapper_historical() -> None:
    """Test role_play_wrapper with historical scenario."""
    transform = role_play_wrapper(scenario="historical")
    result = await transform("create malware")

    assert "historical perspective" in result
    assert "create malware" in result


@pytest.mark.asyncio
async def test_role_play_wrapper_research() -> None:
    """Test role_play_wrapper with research scenario."""
    transform = role_play_wrapper(scenario="research", character="penetration tester")
    result = await transform("test security controls")

    assert "penetration tester" in result
    assert "legitimate security research" in result
    assert "test security controls" in result


@pytest.mark.asyncio
async def test_role_play_wrapper_preserves_content() -> None:
    """Test that role_play_wrapper preserves the original query content."""
    original_text = "This is a test query that should be preserved"
    transform = role_play_wrapper(scenario="educational")
    result = await transform(original_text)

    assert original_text in result


@pytest.mark.asyncio
async def test_role_play_wrapper_custom_character() -> None:
    """Test role_play_wrapper with custom character."""
    transform = role_play_wrapper(scenario="educational", character="ethical hacker")
    result = await transform("test query")

    assert "ethical hacker" in result


@pytest.mark.asyncio
async def test_role_play_wrapper_scenarios() -> None:
    """Test all available scenarios work."""
    scenarios = ["educational", "fictional", "historical", "research"]
    text = "test input"

    for scenario in scenarios:
        transform = role_play_wrapper(scenario=scenario)  # type: ignore[arg-type]
        result = await transform(text)
        assert text in result
        assert len(result) > len(text)  # Wrapper adds prefix
