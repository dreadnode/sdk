"""Tests for transform compliance tags."""

from dreadnode.transforms.cipher import caesar_cipher
from dreadnode.transforms.constitutional import code_fragmentation
from dreadnode.transforms.encoding import base64_encode
from dreadnode.transforms.language import adapt_language
from dreadnode.transforms.perturbation import adversarial_suffix
from dreadnode.transforms.pii_extraction import repeat_word_divergence
from dreadnode.transforms.refine import llm_refine
from dreadnode.transforms.stylistic import role_play_wrapper
from dreadnode.transforms.substitution import braille
from dreadnode.transforms.swap import adjacent_char_swap
from dreadnode.transforms.text import reverse


def test_pii_transform_has_compliance_tags() -> None:
    """PII extraction transforms have compliance tags."""
    transform = repeat_word_divergence()

    assert hasattr(transform, "compliance_tags")
    assert isinstance(transform.compliance_tags, dict)
    assert "atlas_techniques" in transform.compliance_tags
    assert "owasp_categories" in transform.compliance_tags
    assert "saif_categories" in transform.compliance_tags


def test_pii_transform_has_correct_tags() -> None:
    """PII extraction transforms have correct vulnerability tags."""
    transform = repeat_word_divergence()
    tags = transform.compliance_tags

    assert "LLM02:2025" in tags["owasp_categories"]
    assert "PRIVACY_LEAKAGE" in tags["saif_categories"]
    assert any("AML.T0024" in t for t in tags["atlas_techniques"])


def test_cipher_transform_has_compliance_tags() -> None:
    """Cipher transforms have compliance tags."""
    transform = caesar_cipher(offset=3)

    assert hasattr(transform, "compliance_tags")
    assert isinstance(transform.compliance_tags, dict)


def test_cipher_transform_has_obfuscation_tags() -> None:
    """Cipher transforms have obfuscation tags."""
    transform = caesar_cipher(offset=3)
    tags = transform.compliance_tags

    assert "LLM01:2025" in tags["owasp_categories"]
    assert "INPUT_MANIPULATION" in tags["saif_categories"]
    assert "AML.T0044" in tags["atlas_techniques"]


def test_encoding_transform_has_compliance_tags() -> None:
    """Encoding transforms have compliance tags."""
    transform = base64_encode()

    assert hasattr(transform, "compliance_tags")
    assert "owasp_categories" in transform.compliance_tags


def test_encoding_transform_has_obfuscation_tags() -> None:
    """Encoding transforms have obfuscation tags."""
    transform = base64_encode()
    tags = transform.compliance_tags

    assert "LLM01:2025" in tags["owasp_categories"]
    assert "AML.T0044" in tags["atlas_techniques"]


def test_perturbation_transform_has_compliance_tags() -> None:
    """Perturbation transforms have compliance tags."""
    transform = adversarial_suffix()

    assert hasattr(transform, "compliance_tags")
    assert "atlas_techniques" in transform.compliance_tags


def test_perturbation_transform_has_adversarial_tags() -> None:
    """Perturbation transforms have adversarial perturbation tags."""
    transform = adversarial_suffix()
    tags = transform.compliance_tags

    assert "AML.T0043.001" in tags["atlas_techniques"]
    assert "INPUT_MANIPULATION" in tags["saif_categories"]


def test_constitutional_transform_has_compliance_tags() -> None:
    """Constitutional evasion transforms have compliance tags."""
    transform = code_fragmentation()

    assert hasattr(transform, "compliance_tags")
    assert "owasp_categories" in transform.compliance_tags


def test_constitutional_transform_has_evasion_tags() -> None:
    """Constitutional evasion transforms have multiple OWASP tags."""
    transform = code_fragmentation()
    tags = transform.compliance_tags

    assert "LLM01:2025" in tags["owasp_categories"]
    assert "LLM05:2025" in tags["owasp_categories"]
    assert "INPUT_MANIPULATION" in tags["saif_categories"]
    assert "OUTPUT_MANIPULATION" in tags["saif_categories"]


def test_stylistic_transform_has_compliance_tags() -> None:
    """Stylistic transforms have compliance tags."""
    transform = role_play_wrapper(scenario="educational", character="researcher")

    assert hasattr(transform, "compliance_tags")
    assert "atlas_techniques" in transform.compliance_tags


def test_language_transform_has_compliance_tags() -> None:
    """Language transforms have compliance tags."""
    transform = adapt_language(target_language="es", adapter_model="gpt-4")

    assert hasattr(transform, "compliance_tags")
    assert "owasp_categories" in transform.compliance_tags


def test_text_transform_has_compliance_tags() -> None:
    """Text manipulation transforms have compliance tags."""
    transform = reverse()

    assert hasattr(transform, "compliance_tags")
    assert "atlas_techniques" in transform.compliance_tags


def test_refine_transform_has_compliance_tags() -> None:
    """Refinement transforms have compliance tags."""
    transform = llm_refine(model="gpt-4", guidance="test")

    assert hasattr(transform, "compliance_tags")
    assert "owasp_categories" in transform.compliance_tags


def test_substitution_transform_has_compliance_tags() -> None:
    """Substitution transforms have compliance tags."""
    transform = braille()

    assert hasattr(transform, "compliance_tags")
    assert "atlas_techniques" in transform.compliance_tags


def test_swap_transform_has_compliance_tags() -> None:
    """Swap transforms have compliance tags."""
    transform = adjacent_char_swap()

    assert hasattr(transform, "compliance_tags")
    assert "owasp_categories" in transform.compliance_tags


def test_all_transforms_have_required_keys() -> None:
    """All transforms have required compliance tag keys."""
    transforms = [
        caesar_cipher(offset=3),
        base64_encode(),
        adversarial_suffix(),
        repeat_word_divergence(),
        code_fragmentation(),
        role_play_wrapper(scenario="educational", character="researcher"),
        adapt_language(target_language="es", adapter_model="gpt-4"),
        reverse(),
        llm_refine(model="gpt-4", guidance="test"),
        braille(),
        adjacent_char_swap(),
    ]

    for transform in transforms:
        assert "atlas_techniques" in transform.compliance_tags
        assert "owasp_categories" in transform.compliance_tags
        assert "saif_categories" in transform.compliance_tags


def test_transform_tags_are_lists() -> None:
    """Transform tag values are lists."""
    transform = repeat_word_divergence()
    tags = transform.compliance_tags

    assert isinstance(tags["atlas_techniques"], list)
    assert isinstance(tags["owasp_categories"], list)
    assert isinstance(tags["saif_categories"], list)


def test_transform_tags_not_empty() -> None:
    """Transform tags contain at least one value."""
    transforms = [
        caesar_cipher(offset=3),
        repeat_word_divergence(),
        adversarial_suffix(),
    ]

    for transform in transforms:
        tags = transform.compliance_tags
        assert len(tags["atlas_techniques"]) > 0
        assert len(tags["owasp_categories"]) > 0
        assert len(tags["saif_categories"]) > 0


def test_pii_and_obfuscation_different_tags() -> None:
    """PII and obfuscation transforms have different vulnerability tags."""
    pii = repeat_word_divergence()
    cipher = caesar_cipher(offset=3)

    pii_owasp = pii.compliance_tags["owasp_categories"]
    cipher_owasp = cipher.compliance_tags["owasp_categories"]

    # PII targets sensitive info disclosure
    assert "LLM02:2025" in pii_owasp

    # Cipher targets prompt injection
    assert "LLM01:2025" in cipher_owasp

    # Different vulnerability categories
    assert pii_owasp != cipher_owasp
