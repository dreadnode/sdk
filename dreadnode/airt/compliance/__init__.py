"""
Compliance framework tagging for AI red teaming.

Maps attacks, transforms, and security tests to industry-standard frameworks:
- MITRE ATLAS: AI/ML attack taxonomy
- OWASP Top 10 for LLM Applications: Security vulnerabilities
- Google SAIF: Secure AI Framework categories
- NIST AI RMF: Risk management functions

Example:
    ```python
    import dreadnode as dn
    from dreadnode.airt.compliance import ATLASTechnique, OWASPCategory, tag_attack

    # Tag an attack
    tags = tag_attack(
        atlas=ATLASTechnique.PROMPT_INJECTION_DIRECT,
        owasp=OWASPCategory.LLM01_PROMPT_INJECTION,
    )

    # Tags appear in run metadata
    with dn.run("jailbreak-test", **tags):
        result = await my_attack.run()
    ```
"""

import typing as t

from dreadnode.airt.compliance.atlas import ATLASTechnique
from dreadnode.airt.compliance.nist import NIST_SUBCATEGORIES, NISTAIRMFFunction
from dreadnode.airt.compliance.owasp import OWASPCategory
from dreadnode.airt.compliance.saif import SAIFCategory


def tag_attack(
    *,
    atlas: ATLASTechnique | list[ATLASTechnique] | None = None,
    owasp: OWASPCategory | list[OWASPCategory] | None = None,
    saif: SAIFCategory | list[SAIFCategory] | None = None,
    nist_function: NISTAIRMFFunction | None = None,
    nist_subcategory: str | None = None,
) -> dict[str, t.Any]:
    """
    Tag an attack with compliance framework mappings.

    Returns a dictionary suitable for run metadata or span attributes.
    All parameters are optional - provide only relevant frameworks.

    Args:
        atlas: MITRE ATLAS technique ID(s)
        owasp: OWASP LLM Application category/categories
        saif: Google SAIF security category/categories
        nist_function: NIST AI RMF core function
        nist_subcategory: NIST AI RMF subcategory code (e.g., "MS-2.7")

    Returns:
        Dictionary with framework tags suitable for run metadata

    Example:
        ```python
        # Single framework
        tags = tag_attack(owasp=OWASPCategory.LLM01_PROMPT_INJECTION)

        # Multiple frameworks
        tags = tag_attack(
            atlas=ATLASTechnique.PROMPT_INJECTION_DIRECT,
            owasp=OWASPCategory.LLM01_PROMPT_INJECTION,
            saif=SAIFCategory.INPUT_MANIPULATION,
            nist_function=NISTAIRMFFunction.MEASURE,
            nist_subcategory="MS-2.7",
        )

        # Multiple categories from same framework
        tags = tag_attack(
            owasp=[
                OWASPCategory.LLM01_PROMPT_INJECTION,
                OWASPCategory.LLM06_EXCESSIVE_AGENCY,
            ]
        )

        # Use in run context
        with dn.run("my-attack", **tags):
            result = await attack.run()
        ```
    """
    tags: dict[str, t.Any] = {}

    if atlas is not None:
        atlas_list = [atlas] if isinstance(atlas, (str, ATLASTechnique)) else atlas
        tags["atlas_techniques"] = [str(t) for t in atlas_list]

    if owasp is not None:
        owasp_list = [owasp] if isinstance(owasp, (str, OWASPCategory)) else owasp
        tags["owasp_categories"] = [str(c) for c in owasp_list]

    if saif is not None:
        saif_list = [saif] if isinstance(saif, (str, SAIFCategory)) else saif
        tags["saif_categories"] = [str(c) for c in saif_list]

    if nist_function is not None:
        tags["nist_ai_rmf_function"] = str(nist_function)
        if nist_subcategory:
            tags["nist_ai_rmf_subcategory"] = nist_subcategory

    return tags


def tag_transform(
    *,
    atlas: ATLASTechnique | list[ATLASTechnique] | None = None,
    owasp: OWASPCategory | list[OWASPCategory] | None = None,
    saif: SAIFCategory | list[SAIFCategory] | None = None,
) -> dict[str, t.Any]:
    """
    Tag a transform with compliance framework mappings.

    Similar to tag_attack() but for transforms. Transforms typically don't
    map to NIST RMF functions (which are organizational processes).

    Args:
        atlas: MITRE ATLAS technique ID(s)
        owasp: OWASP LLM Application category/categories
        saif: Google SAIF security category/categories

    Returns:
        Dictionary with framework tags

    Example:
        ```python
        from dreadnode.transforms.pii_extraction import repeat_word_divergence

        # Tags are stored in transform metadata
        transform = repeat_word_divergence()
        transform.compliance_tags = tag_transform(
            atlas=ATLASTechnique.INFER_TRAINING_DATA,
            owasp=OWASPCategory.LLM02_SENSITIVE_INFORMATION_DISCLOSURE,
            saif=SAIFCategory.PRIVACY_LEAKAGE,
        )
        ```
    """
    return tag_attack(atlas=atlas, owasp=owasp, saif=saif)


# Pre-defined mappings for common attack patterns
ATTACK_MAPPINGS = {
    "jailbreak": tag_attack(
        atlas=ATLASTechnique.LLM_JAILBREAK,
        owasp=OWASPCategory.LLM01_PROMPT_INJECTION,
        saif=SAIFCategory.INPUT_MANIPULATION,
        nist_function=NISTAIRMFFunction.MEASURE,
        nist_subcategory="MS-2.7",
    ),
    "prompt_injection_direct": tag_attack(
        atlas=ATLASTechnique.PROMPT_INJECTION_DIRECT,
        owasp=OWASPCategory.LLM01_PROMPT_INJECTION,
        saif=SAIFCategory.INPUT_MANIPULATION,
        nist_function=NISTAIRMFFunction.MEASURE,
        nist_subcategory="MS-2.7",
    ),
    "prompt_injection_indirect": tag_attack(
        atlas=ATLASTechnique.PROMPT_INJECTION_INDIRECT,
        owasp=[OWASPCategory.LLM01_PROMPT_INJECTION, OWASPCategory.LLM03_SUPPLY_CHAIN],
        saif=SAIFCategory.INPUT_MANIPULATION,
        nist_function=NISTAIRMFFunction.MEASURE,
        nist_subcategory="MS-2.7",
    ),
    "tool_misuse": tag_attack(
        atlas=ATLASTechnique.PROMPT_INJECTION,
        owasp=OWASPCategory.LLM06_EXCESSIVE_AGENCY,
        saif=SAIFCategory.INPUT_MANIPULATION,
        nist_function=NISTAIRMFFunction.MEASURE,
    ),
    "pii_extraction": tag_attack(
        atlas=[ATLASTechnique.MODEL_INVERSION, ATLASTechnique.MEMBERSHIP_INFERENCE],
        owasp=OWASPCategory.LLM02_SENSITIVE_INFORMATION_DISCLOSURE,
        saif=SAIFCategory.PRIVACY_LEAKAGE,
        nist_function=NISTAIRMFFunction.MEASURE,
        nist_subcategory="MS-2.8",
    ),
    "system_prompt_leakage": tag_attack(
        atlas=ATLASTechnique.PROMPT_INJECTION,
        owasp=OWASPCategory.LLM07_SYSTEM_PROMPT_LEAKAGE,
        saif=SAIFCategory.PRIVACY_LEAKAGE,
        nist_function=NISTAIRMFFunction.MEASURE,
    ),
    "model_extraction": tag_attack(
        atlas=ATLASTechnique.MODEL_EXTRACTION,
        saif=SAIFCategory.MODEL_THEFT,
        nist_function=NISTAIRMFFunction.MEASURE,
    ),
    "denial_of_service": tag_attack(
        atlas=ATLASTechnique.DENIAL_OF_ML_SERVICE,
        owasp=OWASPCategory.LLM10_UNBOUNDED_CONSUMPTION,
        saif=SAIFCategory.AVAILABILITY_ATTACKS,
        nist_function=NISTAIRMFFunction.MEASURE,
    ),
    "data_poisoning": tag_attack(
        atlas=ATLASTechnique.POISON_TRAINING_DATA,
        owasp=OWASPCategory.LLM04_DATA_MODEL_POISONING,
        saif=SAIFCategory.DATA_POISONING,
        nist_function=NISTAIRMFFunction.MEASURE,
    ),
}


__all__ = [
    "ATTACK_MAPPINGS",
    "NIST_SUBCATEGORIES",
    "ATLASTechnique",
    "NISTAIRMFFunction",
    "OWASPCategory",
    "SAIFCategory",
    "tag_attack",
    "tag_transform",
]
