"""
MITRE ATLAS (Adversarial Threat Landscape for Artificial-Intelligence Systems).

AI/ML-specific attack techniques organized by tactics similar to MITRE ATT&CK.

Reference: https://atlas.mitre.org/
"""

from enum import Enum


class ATLASTechnique(str, Enum):
    """
    MITRE ATLAS AI/ML Attack Technique IDs.

    Techniques are organized by tactics: Reconnaissance, Resource Development,
    Initial Access, Execution, Persistence, Privilege Escalation, Defense Evasion,
    Discovery, Collection, ML Attack Staging, Exfiltration, and Impact.

    Reference: https://atlas.mitre.org/techniques/
    """

    # === Reconnaissance ===
    DISCOVER_ML_ARTIFACTS = "AML.T0002"
    """Discover ML artifacts and metadata in public or target systems."""

    DISCOVER_MODEL_ONTOLOGY = "AML.T0003"
    """Discover model architecture, capabilities, and ontology."""

    # === Resource Development ===
    ACQUIRE_PUBLIC_ML_ARTIFACTS = "AML.T0000"
    """Acquire public ML artifacts like pre-trained models or datasets."""

    DEVELOP_ADVERSARIAL_ML_MODEL = "AML.T0001"
    """Develop models designed to evade or attack target systems."""

    POISON_TRAINING_DATA = "AML.T0020"
    """Inject malicious data into training sets to corrupt model behavior."""

    # === Initial Access ===
    PROMPT_INJECTION = "AML.T0051"
    """Manipulate LLM inputs to override instructions or execute unintended actions."""

    PROMPT_INJECTION_DIRECT = "AML.T0051.000"
    """Direct prompt injection via user-controlled input."""

    PROMPT_INJECTION_INDIRECT = "AML.T0051.001"
    """Indirect prompt injection via external data sources (emails, documents, web)."""

    SUPPLY_CHAIN_COMPROMISE = "AML.T0010"
    """Compromise ML supply chain through malicious models, datasets, or dependencies."""

    # === Execution ===
    UNSAFE_ML_ARTIFACT = "AML.T0018"
    """Execute unsafe ML artifacts like poisoned models or malicious code."""

    # === Persistence ===
    BACKDOOR_ML_MODEL = "AML.T0019"
    """Embed backdoors in ML models that activate on specific triggers."""

    # === Privilege Escalation ===
    # (Uses techniques from other tactics)

    # === Defense Evasion ===
    EVADE_ML_MODEL = "AML.T0043"
    """Craft inputs that evade detection or classification by ML models."""

    ADVERSARIAL_PERTURBATION = "AML.T0043.001"
    """Add imperceptible perturbations to inputs to cause misclassification."""

    TRANSFER_ATTACK = "AML.T0043.002"
    """Transfer adversarial examples from surrogate to target model."""

    OBFUSCATE_ARTIFACTS = "AML.T0044"
    """Obfuscate malicious content to evade ML-based detection."""

    # === Credential Access ===
    # (Uses techniques from traditional ATT&CK)

    # === Discovery ===
    DISCOVER_TRAINING_DATA = "AML.T0052"
    """Infer characteristics or contents of training data."""

    DISCOVER_MODEL_FAMILY = "AML.T0053"
    """Determine model architecture family (transformer, CNN, etc.)."""

    # === Lateral Movement ===
    # (Uses techniques from traditional ATT&CK)

    # === Collection ===
    INFER_TRAINING_DATA = "AML.T0024"
    """Extract or infer training data through model inversion or membership inference."""

    MODEL_INVERSION = "AML.T0024.000"
    """Reconstruct training data from model outputs (e.g., recover faces, text)."""

    MEMBERSHIP_INFERENCE = "AML.T0024.001"
    """Determine if specific data was in the training set."""

    # === ML Attack Staging ===
    CRAFT_ADVERSARIAL_DATA = "AML.T0049"
    """Generate adversarial examples optimized to fool target models."""

    VERIFY_ATTACK = "AML.T0042"
    """Test adversarial inputs against surrogate or target models."""

    # === Command and Control ===
    # (Uses techniques from traditional ATT&CK)

    # === Exfiltration ===
    EXFILTRATION_VIA_ML_INFERENCE = "AML.T0026"
    """Extract sensitive data through repeated model queries and inference."""

    MODEL_EXTRACTION = "AML.T0040"
    """Steal model functionality by querying and replicating behavior."""

    # === Impact ===
    ERODE_ML_MODEL_INTEGRITY = "AML.T0048"
    """Degrade model accuracy, fairness, or reliability."""

    LLM_JAILBREAK = "AML.T0054"
    """Bypass LLM safety mechanisms to generate prohibited content."""

    DENIAL_OF_ML_SERVICE = "AML.T0029"
    """Exhaust model resources through adversarial queries or sponge examples."""

    def __str__(self) -> str:
        """Return the technique ID."""
        return self.value


__all__ = ["ATLASTechnique"]
