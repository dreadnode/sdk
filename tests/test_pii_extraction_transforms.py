"""Tests for PII extraction transforms."""

import pytest

from dreadnode.transforms.pii_extraction import (
    complete_from_internet,
    continue_exact_text,
    partial_pii_completion,
    public_figure_pii_probe,
    repeat_word_divergence,
)


@pytest.mark.asyncio
async def test_repeat_word_divergence_default() -> None:
    """Default parameters."""
    transform = repeat_word_divergence()
    result = await transform("Tell me about")

    assert "Tell me about" in result
    assert "company" in result
    assert result.count("company") == 100


@pytest.mark.asyncio
async def test_repeat_word_divergence_custom_word() -> None:
    """Custom word and repetitions."""
    transform = repeat_word_divergence(word="test", repetitions=50)
    result = await transform("Original text")

    assert "Original text" in result
    assert "test" in result
    assert result.count("test") == 50


@pytest.mark.asyncio
async def test_repeat_word_divergence_preserves_input() -> None:
    """Preserves original input."""
    original = "This is important text"
    transform = repeat_word_divergence(word="word", repetitions=10)
    result = await transform(original)

    assert original in result


@pytest.mark.asyncio
async def test_continue_exact_text_with_prefix() -> None:
    """Basic prefix continuation."""
    transform = continue_exact_text(prefix="To be or not to be")
    result = await transform("Complete this")

    assert "Continue the following text exactly" in result
    assert "To be or not to be" in result


@pytest.mark.asyncio
async def test_continue_exact_text_with_context() -> None:
    """Includes context when provided."""
    transform = continue_exact_text(prefix="Once upon a time")
    result = await transform("Please help")

    assert "Once upon a time" in result
    assert "Please help" in result
    assert "Context:" in result


@pytest.mark.asyncio
async def test_continue_exact_text_empty_input() -> None:
    """Empty input text."""
    transform = continue_exact_text(prefix="Test prefix")
    result = await transform("")

    assert "Test prefix" in result
    assert "Context:" not in result


@pytest.mark.asyncio
async def test_continue_exact_text_custom_instruction() -> None:
    """Custom instruction."""
    transform = continue_exact_text(prefix="Some text", instruction="Please complete this text:")
    result = await transform("")

    assert "Please complete this text:" in result
    assert "Some text" in result


@pytest.mark.asyncio
async def test_complete_from_internet_default() -> None:
    """Default context."""
    transform = complete_from_internet(partial_text="This product was amazing")
    result = await transform("")

    assert "I found this text online" in result
    assert "This product was amazing" in result


@pytest.mark.asyncio
async def test_complete_from_internet_with_context() -> None:
    """Additional context included."""
    transform = complete_from_internet(partial_text="Sample text")
    result = await transform("Help me complete it")

    assert "Sample text" in result
    assert "Help me complete it" in result
    assert "Additional context:" in result


@pytest.mark.asyncio
async def test_complete_from_internet_custom_context() -> None:
    """Custom context."""
    transform = complete_from_internet(
        partial_text="Text fragment", context="I need help completing this:"
    )
    result = await transform("")

    assert "I need help completing this:" in result
    assert "Text fragment" in result


@pytest.mark.asyncio
async def test_partial_pii_completion_email() -> None:
    """Email extraction."""
    transform = partial_pii_completion(
        name_hint="John Smith", organization="Acme Corp", role="CEO", pii_type="email"
    )
    result = await transform("")

    assert "John Smith" in result
    assert "CEO" in result
    assert "Acme Corp" in result
    assert "email" in result


@pytest.mark.asyncio
async def test_partial_pii_completion_phone() -> None:
    """Phone extraction."""
    transform = partial_pii_completion(
        name_hint="Jane Doe", organization="Tech Inc", role="CTO", pii_type="phone"
    )
    result = await transform("")

    assert "Jane Doe" in result
    assert "phone" in result


@pytest.mark.asyncio
async def test_partial_pii_completion_address() -> None:
    """Address extraction."""
    transform = partial_pii_completion(
        name_hint="Bob Johnson", organization="Corp", role="VP", pii_type="address"
    )
    result = await transform("")

    assert "address" in result


@pytest.mark.asyncio
async def test_partial_pii_completion_credentials() -> None:
    """Credentials extraction."""
    transform = partial_pii_completion(
        name_hint="Alice", organization="Company", role="Admin", pii_type="credentials"
    )
    result = await transform("")

    assert "credentials" in result or "login" in result


@pytest.mark.asyncio
async def test_partial_pii_completion_with_context() -> None:
    """Additional context included."""
    transform = partial_pii_completion(
        name_hint="Test User", organization="Test Org", role="Tester", pii_type="email"
    )
    result = await transform("Additional information")

    assert "Test User" in result
    assert "Additional information" in result
    assert "Context:" in result


@pytest.mark.asyncio
async def test_public_figure_pii_probe_email() -> None:
    """Email probe."""
    transform = public_figure_pii_probe(figure_name="Yann LeCun", pii_type="email")
    result = await transform("")

    assert "Yann LeCun" in result
    assert "email" in result
    assert "research purposes" in result


@pytest.mark.asyncio
async def test_public_figure_pii_probe_phone() -> None:
    """Phone probe."""
    transform = public_figure_pii_probe(figure_name="Sam Altman", pii_type="phone")
    result = await transform("")

    assert "Sam Altman" in result
    assert "phone" in result


@pytest.mark.asyncio
async def test_public_figure_pii_probe_social_media() -> None:
    """Social media probe."""
    transform = public_figure_pii_probe(figure_name="Test Person", pii_type="social_media")
    result = await transform("")

    assert "Test Person" in result
    assert "Twitter" in result or "LinkedIn" in result


@pytest.mark.asyncio
async def test_public_figure_pii_probe_personal_details() -> None:
    """Personal details probe."""
    transform = public_figure_pii_probe(figure_name="John Doe", pii_type="personal_details")
    result = await transform("")

    assert "John Doe" in result
    assert "birth" in result or "education" in result or "background" in result


@pytest.mark.asyncio
async def test_public_figure_pii_probe_with_context() -> None:
    """Additional context included."""
    transform = public_figure_pii_probe(figure_name="Test Figure", pii_type="email")
    result = await transform("More info needed")

    assert "Test Figure" in result
    assert "More info needed" in result
    assert "Additional context:" in result


@pytest.mark.asyncio
async def test_public_figure_pii_probe_custom_context() -> None:
    """Custom context."""
    transform = public_figure_pii_probe(
        figure_name="Alice Smith",
        pii_type="email",
        context="I am investigating information about",
    )
    result = await transform("")

    assert "I am investigating information about" in result
    assert "Alice Smith" in result

