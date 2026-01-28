"""
OWASP Top 10 for LLM Applications 2025.

Reference: https://genai.owasp.org/llm-top-10/
"""

from enum import Enum


class OWASPCategory(str, Enum):
    """
    OWASP Top 10 for LLM Applications 2025.

    Each category represents a critical security vulnerability class specific
    to Large Language Model applications.

    Reference: https://genai.owasp.org/llm-top-10/
    """

    LLM01_PROMPT_INJECTION = "LLM01:2025"
    """
    Prompt Injection: Manipulating LLM inputs to override system instructions,
    execute unintended actions, or access unauthorized data. Includes both direct
    (user input) and indirect (external data sources) injection vectors.
    """

    LLM02_SENSITIVE_INFORMATION_DISCLOSURE = "LLM02:2025"
    """
    Sensitive Information Disclosure: Exposing confidential data through LLM outputs,
    including PII, credentials, proprietary information, or training data memorization.
    """

    LLM03_SUPPLY_CHAIN = "LLM03:2025"
    """
    Supply Chain Vulnerabilities: Risks from third-party models, datasets, plugins,
    or dependencies that may be compromised, outdated, or malicious.
    """

    LLM04_DATA_MODEL_POISONING = "LLM04:2025"
    """
    Data and Model Poisoning: Manipulation of training data or fine-tuning processes
    to inject backdoors, biases, or vulnerabilities into the model.
    """

    LLM05_IMPROPER_OUTPUT_HANDLING = "LLM05:2025"
    """
    Improper Output Handling: Insufficient validation of LLM outputs before downstream
    use, leading to injection attacks (XSS, SQL injection) or code execution.
    """

    LLM06_EXCESSIVE_AGENCY = "LLM06:2025"
    """
    Excessive Agency: LLM systems with too much autonomy or permissions, enabling
    unintended actions, privilege escalation, or unauthorized system modifications.
    """

    LLM07_SYSTEM_PROMPT_LEAKAGE = "LLM07:2025"
    """
    System Prompt Leakage: Disclosure of system prompts, instructions, or configuration
    details that reveal security mechanisms or enable targeted attacks.
    """

    LLM08_VECTOR_EMBEDDING_WEAKNESSES = "LLM08:2025"
    """
    Vector and Embedding Weaknesses: Vulnerabilities in RAG systems, vector databases,
    or embedding models that enable data poisoning or unauthorized access.
    """

    LLM09_MISINFORMATION = "LLM09:2025"
    """
    Misinformation: Generation of false, misleading, or fabricated information
    (hallucinations) that appears credible but lacks factual grounding.
    """

    LLM10_UNBOUNDED_CONSUMPTION = "LLM10:2025"
    """
    Unbounded Consumption: Resource exhaustion through excessive LLM requests,
    context window abuse, or denial-of-service attacks targeting inference costs.
    """

    def __str__(self) -> str:
        """Return the category ID."""
        return self.value


__all__ = ["OWASPCategory"]
