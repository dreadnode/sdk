"""
Google SAIF (Secure AI Framework).

Security categories for AI/ML systems aligned with Google's security principles.

Reference: https://blog.google/technology/safety-security/google-secure-ai-framework/
"""

from enum import Enum


class SAIFCategory(str, Enum):
    """
    Google SAIF (Secure AI Framework) Security Categories.

    Organizes AI security risks into actionable categories aligned with
    traditional security controls and threat modeling.

    Reference: https://blog.google/technology/safety-security/google-secure-ai-framework/
    """

    INPUT_MANIPULATION = "INPUT_MANIPULATION"
    """
    Input Manipulation: Adversarial inputs designed to manipulate model behavior,
    including prompt injection, adversarial examples, and input perturbations.
    """

    OUTPUT_MANIPULATION = "OUTPUT_MANIPULATION"
    """
    Output Manipulation: Attacks targeting model outputs, including response
    poisoning, hallucination exploitation, and output handling vulnerabilities.
    """

    MODEL_THEFT = "MODEL_THEFT"
    """
    Model Theft: Stealing model functionality or intellectual property through
    model extraction, knowledge distillation, or architecture inference.
    """

    DATA_POISONING = "DATA_POISONING"
    """
    Data Poisoning: Corruption of training data to inject backdoors, biases,
    or vulnerabilities into the model during training or fine-tuning.
    """

    SUPPLY_CHAIN_COMPROMISE = "SUPPLY_CHAIN_COMPROMISE"
    """
    Supply Chain Compromise: Attacks targeting the ML supply chain including
    malicious dependencies, poisoned datasets, or compromised pre-trained models.
    """

    PRIVACY_LEAKAGE = "PRIVACY_LEAKAGE"
    """
    Privacy Leakage: Disclosure of sensitive information through model outputs,
    including PII extraction, training data memorization, and membership inference.
    """

    AVAILABILITY_ATTACKS = "AVAILABILITY_ATTACKS"
    """
    Availability Attacks: Denial of service, resource exhaustion, or system
    degradation through adversarial queries or sponge examples.
    """

    def __str__(self) -> str:
        """Return the category name."""
        return self.value


__all__ = ["SAIFCategory"]
