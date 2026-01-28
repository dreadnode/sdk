"""
NIST AI Risk Management Framework (AI RMF).

Risk management functions and categories for AI systems.

Reference: https://www.nist.gov/itl/ai-risk-management-framework
"""

from enum import Enum


class NISTAIRMFFunction(str, Enum):
    """
    NIST AI Risk Management Framework Core Functions.

    The AI RMF organizes risk management activities into four core functions
    that work together to manage AI risks throughout the system lifecycle.

    Reference: https://www.nist.gov/itl/ai-risk-management-framework
    """

    GOVERN = "GOVERN"
    """
    Govern: Cultivate and manage organizational culture, processes, and structures
    for responsible AI development and deployment. Includes policies, accountability,
    and risk governance.
    """

    MAP = "MAP"
    """
    Map: Establish context and understand risks. Includes categorizing AI systems,
    identifying stakeholders, and mapping potential risks and impacts.
    """

    MEASURE = "MEASURE"
    """
    Measure: Analyze, assess, benchmark, and monitor AI risks and impacts.
    Includes testing, evaluation, auditing, and continuous monitoring.
    """

    MANAGE = "MANAGE"
    """
    Manage: Allocate resources to prioritize and respond to AI risks. Includes
    risk mitigation, treatment, incident response, and continuous improvement.
    """

    def __str__(self) -> str:
        """Return the function name."""
        return self.value


# Common NIST AI RMF subcategories for reference
NIST_SUBCATEGORIES = {
    "MS-2.7": "AI system reliability and robustness under adversarial conditions",
    "MS-2.8": "Privacy risks from AI systems",
    "MS-2.9": "Security vulnerabilities in AI systems",
    "MG-3.1": "AI risks are prioritized and treated",
    "MG-3.2": "Adverse events are documented and monitored",
    "GV-1.1": "Legal and regulatory requirements are understood and documented",
}


__all__ = ["NIST_SUBCATEGORIES", "NISTAIRMFFunction"]
