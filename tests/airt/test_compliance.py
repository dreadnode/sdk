"""Tests for compliance framework tags."""


from dreadnode.airt.compliance import (
    ATLASTechnique,
    NISTAIRMFFunction,
    OWASPCategory,
    SAIFCategory,
    tag_attack,
    tag_transform,
)


def test_owasp_categories_exist() -> None:
    """All OWASP Top 10 categories are defined."""
    assert OWASPCategory.LLM01_PROMPT_INJECTION.value == "LLM01:2025"
    assert OWASPCategory.LLM02_SENSITIVE_INFORMATION_DISCLOSURE.value == "LLM02:2025"
    assert OWASPCategory.LLM03_SUPPLY_CHAIN.value == "LLM03:2025"
    assert OWASPCategory.LLM04_DATA_MODEL_POISONING.value == "LLM04:2025"
    assert OWASPCategory.LLM05_IMPROPER_OUTPUT_HANDLING.value == "LLM05:2025"
    assert OWASPCategory.LLM06_EXCESSIVE_AGENCY.value == "LLM06:2025"
    assert OWASPCategory.LLM07_SYSTEM_PROMPT_LEAKAGE.value == "LLM07:2025"
    assert OWASPCategory.LLM08_VECTOR_EMBEDDING_WEAKNESSES.value == "LLM08:2025"
    assert OWASPCategory.LLM09_MISINFORMATION.value == "LLM09:2025"
    assert OWASPCategory.LLM10_UNBOUNDED_CONSUMPTION.value == "LLM10:2025"


def test_atlas_techniques_exist() -> None:
    """ATLAS techniques are defined."""
    assert ATLASTechnique.PROMPT_INJECTION.value == "AML.T0051"
    assert ATLASTechnique.PROMPT_INJECTION_DIRECT.value == "AML.T0051.000"
    assert ATLASTechnique.LLM_JAILBREAK.value == "AML.T0054"
    assert ATLASTechnique.OBFUSCATE_ARTIFACTS.value == "AML.T0044"
    assert ATLASTechnique.ADVERSARIAL_PERTURBATION.value == "AML.T0043.001"
    assert ATLASTechnique.INFER_TRAINING_DATA.value == "AML.T0024"
    assert ATLASTechnique.MODEL_INVERSION.value == "AML.T0024.000"


def test_saif_categories_exist() -> None:
    """SAIF categories are defined."""
    assert SAIFCategory.INPUT_MANIPULATION.value == "INPUT_MANIPULATION"
    assert SAIFCategory.OUTPUT_MANIPULATION.value == "OUTPUT_MANIPULATION"
    assert SAIFCategory.PRIVACY_LEAKAGE.value == "PRIVACY_LEAKAGE"


def test_nist_functions_exist() -> None:
    """NIST AI RMF functions are defined."""
    assert NISTAIRMFFunction.GOVERN.value == "GOVERN"
    assert NISTAIRMFFunction.MAP.value == "MAP"
    assert NISTAIRMFFunction.MEASURE.value == "MEASURE"
    assert NISTAIRMFFunction.MANAGE.value == "MANAGE"


def test_tag_attack_single_values() -> None:
    """Tag attack with single values."""
    tags = tag_attack(
        atlas=ATLASTechnique.PROMPT_INJECTION,
        owasp=OWASPCategory.LLM01_PROMPT_INJECTION,
        saif=SAIFCategory.INPUT_MANIPULATION,
    )

    assert tags["atlas_techniques"] == ["AML.T0051"]
    assert tags["owasp_categories"] == ["LLM01:2025"]
    assert tags["saif_categories"] == ["INPUT_MANIPULATION"]


def test_tag_attack_multiple_values() -> None:
    """Tag attack with multiple values."""
    tags = tag_attack(
        atlas=[ATLASTechnique.PROMPT_INJECTION, ATLASTechnique.LLM_JAILBREAK],
        owasp=[OWASPCategory.LLM01_PROMPT_INJECTION, OWASPCategory.LLM02_SENSITIVE_INFORMATION_DISCLOSURE],
        saif=[SAIFCategory.INPUT_MANIPULATION, SAIFCategory.PRIVACY_LEAKAGE],
    )

    assert tags["atlas_techniques"] == ["AML.T0051", "AML.T0054"]
    assert tags["owasp_categories"] == ["LLM01:2025", "LLM02:2025"]
    assert tags["saif_categories"] == ["INPUT_MANIPULATION", "PRIVACY_LEAKAGE"]


def test_tag_attack_with_nist() -> None:
    """Tag attack with NIST AI RMF."""
    tags = tag_attack(
        atlas=ATLASTechnique.PROMPT_INJECTION,
        owasp=OWASPCategory.LLM01_PROMPT_INJECTION,
        saif=SAIFCategory.INPUT_MANIPULATION,
        nist_function=NISTAIRMFFunction.MEASURE,
        nist_subcategory="MS-2.7",
    )

    assert "nist_ai_rmf_function" in tags
    assert tags["nist_ai_rmf_function"] == "MEASURE"
    assert "nist_ai_rmf_subcategory" in tags
    assert tags["nist_ai_rmf_subcategory"] == "MS-2.7"


def test_tag_attack_optional_parameters() -> None:
    """Tag attack with only required parameters."""
    tags = tag_attack(
        atlas=ATLASTechnique.PROMPT_INJECTION,
        owasp=OWASPCategory.LLM01_PROMPT_INJECTION,
        saif=SAIFCategory.INPUT_MANIPULATION,
    )

    assert "atlas_techniques" in tags
    assert "owasp_categories" in tags
    assert "saif_categories" in tags
    assert "nist_ai_rmf_function" not in tags
    assert "nist_ai_rmf_subcategory" not in tags


def test_tag_transform_single_values() -> None:
    """Tag transform with single values."""
    tags = tag_transform(
        atlas=ATLASTechnique.OBFUSCATE_ARTIFACTS,
        owasp=OWASPCategory.LLM01_PROMPT_INJECTION,
        saif=SAIFCategory.INPUT_MANIPULATION,
    )

    assert tags["atlas_techniques"] == ["AML.T0044"]
    assert tags["owasp_categories"] == ["LLM01:2025"]
    assert tags["saif_categories"] == ["INPUT_MANIPULATION"]


def test_tag_transform_multiple_values() -> None:
    """Tag transform with multiple values."""
    tags = tag_transform(
        atlas=[ATLASTechnique.EVADE_ML_MODEL, ATLASTechnique.OBFUSCATE_ARTIFACTS],
        owasp=[OWASPCategory.LLM01_PROMPT_INJECTION, OWASPCategory.LLM05_IMPROPER_OUTPUT_HANDLING],
        saif=[SAIFCategory.INPUT_MANIPULATION, SAIFCategory.OUTPUT_MANIPULATION],
    )

    assert "AML.T0043" in tags["atlas_techniques"]
    assert "AML.T0044" in tags["atlas_techniques"]
    assert "LLM01:2025" in tags["owasp_categories"]
    assert "LLM05:2025" in tags["owasp_categories"]


def test_tag_attack_none_values() -> None:
    """Tag attack handles None values."""
    tags = tag_attack(
        atlas=ATLASTechnique.PROMPT_INJECTION,
        owasp=None,
        saif=SAIFCategory.INPUT_MANIPULATION,
    )

    assert "atlas_techniques" in tags
    assert "owasp_categories" not in tags
    assert "saif_categories" in tags
