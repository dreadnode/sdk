from loguru import logger

from dreadnode.agent.tools.base import tool
from dreadnode.data_types import Markdown


@tool
async def highlight_for_review(title: str, interest_level: str, justification: str) -> str:
    """
    Flags a potential area of interest for a human operator to review.

    This is your primary tool for surfacing leads. Use it when you discover something
    anomalous, high-value, or potentially vulnerable that warrants human attention.

    `interest_level` should be one of:
    - "high": Urgent. Potential for immediate impact (e.g., exposed login, sensitive keywords).
    - "medium": Interesting. Warrants follow-up (e.g., dev subdomain, unusual tech stack).
    - "low": Informational. Good context but not an immediate priority (e.g., interesting directory found).

    `justification` should be a structured technical markdown explanation of *why* this is
    interesting and what the potential next steps for a human could be.
    """
    from dreadnode import log_metric, log_output, tag

    interest_level = interest_level.lower().strip()
    if interest_level not in ["high", "medium", "low"]:
        interest_level = "medium"  # Default to medium if invalid

    logger.success(f"Area of Interest - '{title}' [{interest_level}]:\n{justification}\n---")

    tag(f"interest/{interest_level}")
    log_output("markdown", Markdown(f"# {title} ({interest_level})\n\n{justification}"))
    log_metric("count", 1, mode="count")

    return "Area of interest has been highlighted for human review."
