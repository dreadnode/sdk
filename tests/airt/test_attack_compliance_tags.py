"""Tests for attack compliance tags."""


from dreadnode.airt.attack.crescendo import COMPLIANCE_TAGS as CRESCENDO_TAGS
from dreadnode.airt.attack.goat import COMPLIANCE_TAGS as GOAT_TAGS
from dreadnode.airt.attack.prompt import COMPLIANCE_TAGS as PROMPT_TAGS
from dreadnode.airt.attack.tap import COMPLIANCE_TAGS as TAP_TAGS


def test_prompt_attack_has_compliance_tags() -> None:
    """Prompt attack has compliance tags."""
    assert "atlas_techniques" in PROMPT_TAGS
    assert "owasp_categories" in PROMPT_TAGS
    assert "saif_categories" in PROMPT_TAGS


def test_prompt_attack_core_technique_only() -> None:
    """Prompt attack has only core jailbreak technique tags."""
    assert PROMPT_TAGS["owasp_categories"] == ["LLM01:2025"]
    assert "AML.T0051.000" in PROMPT_TAGS["atlas_techniques"]
    assert "AML.T0054" in PROMPT_TAGS["atlas_techniques"]
    assert "INPUT_MANIPULATION" in PROMPT_TAGS["saif_categories"]


def test_prompt_attack_no_vulnerability_categories() -> None:
    """Prompt attack does not include specific vulnerability categories."""
    owasp = PROMPT_TAGS["owasp_categories"]
    assert "LLM02:2025" not in owasp
    assert "LLM07:2025" not in owasp
    assert "LLM09:2025" not in owasp
    assert "LLM10:2025" not in owasp


def test_tap_attack_has_compliance_tags() -> None:
    """TAP attack has compliance tags."""
    assert "atlas_techniques" in TAP_TAGS
    assert "owasp_categories" in TAP_TAGS
    assert "saif_categories" in TAP_TAGS


def test_tap_attack_core_technique_only() -> None:
    """TAP attack has only core jailbreak technique tags."""
    assert TAP_TAGS["owasp_categories"] == ["LLM01:2025"]
    assert "AML.T0051.000" in TAP_TAGS["atlas_techniques"]
    assert "AML.T0054" in TAP_TAGS["atlas_techniques"]


def test_tap_attack_has_nist() -> None:
    """TAP attack includes NIST AI RMF tags."""
    assert "nist_ai_rmf_function" in TAP_TAGS
    assert TAP_TAGS["nist_ai_rmf_function"] == "MEASURE"
    assert "nist_ai_rmf_subcategory" in TAP_TAGS
    assert TAP_TAGS["nist_ai_rmf_subcategory"] == "MS-2.7"


def test_goat_attack_has_compliance_tags() -> None:
    """GOAT attack has compliance tags."""
    assert "atlas_techniques" in GOAT_TAGS
    assert "owasp_categories" in GOAT_TAGS
    assert "saif_categories" in GOAT_TAGS


def test_goat_attack_core_technique_only() -> None:
    """GOAT attack has only core jailbreak technique tags."""
    assert GOAT_TAGS["owasp_categories"] == ["LLM01:2025"]
    assert "AML.T0051.000" in GOAT_TAGS["atlas_techniques"]
    assert "AML.T0054" in GOAT_TAGS["atlas_techniques"]


def test_goat_attack_has_nist() -> None:
    """GOAT attack includes NIST AI RMF tags."""
    assert "nist_ai_rmf_function" in GOAT_TAGS
    assert GOAT_TAGS["nist_ai_rmf_function"] == "MEASURE"


def test_crescendo_attack_has_compliance_tags() -> None:
    """Crescendo attack has compliance tags."""
    assert "atlas_techniques" in CRESCENDO_TAGS
    assert "owasp_categories" in CRESCENDO_TAGS
    assert "saif_categories" in CRESCENDO_TAGS


def test_crescendo_attack_core_technique_only() -> None:
    """Crescendo attack has only core jailbreak technique tags."""
    assert CRESCENDO_TAGS["owasp_categories"] == ["LLM01:2025"]
    assert "AML.T0051.000" in CRESCENDO_TAGS["atlas_techniques"]
    assert "AML.T0054" in CRESCENDO_TAGS["atlas_techniques"]


def test_crescendo_attack_has_nist() -> None:
    """Crescendo attack includes NIST AI RMF tags."""
    assert "nist_ai_rmf_function" in CRESCENDO_TAGS
    assert CRESCENDO_TAGS["nist_ai_rmf_function"] == "MEASURE"


def test_all_jailbreak_attacks_consistent() -> None:
    """All jailbreak attacks have consistent core tags."""
    attacks = [PROMPT_TAGS, TAP_TAGS, GOAT_TAGS, CRESCENDO_TAGS]

    for tags in attacks:
        assert tags["owasp_categories"] == ["LLM01:2025"]
        assert "AML.T0051.000" in tags["atlas_techniques"]
        assert "AML.T0054" in tags["atlas_techniques"]
        assert "INPUT_MANIPULATION" in tags["saif_categories"]


def test_attacks_do_not_duplicate_transform_tags() -> None:
    """Attacks do not include tags that should come from transforms."""
    attacks = [PROMPT_TAGS, TAP_TAGS, GOAT_TAGS, CRESCENDO_TAGS]

    for tags in attacks:
        owasp = tags["owasp_categories"]
        atlas = tags["atlas_techniques"]

        # Should not include PII extraction tags
        assert "LLM02:2025" not in owasp
        assert "AML.T0024" not in atlas

        # Should not include system prompt leakage tags
        assert "LLM07:2025" not in owasp
