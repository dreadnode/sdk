from loguru import logger

from dreadnode.agent.tools.base import tool
from dreadnode.data_types import Markdown


@tool(catch=True)
async def highlight_for_review(title: str, interest_level: str, justification: str) -> str:
    """
    Flag a finding for human review. Use this to surface leads that warrant further investigation.

    This tool is essential for escalating findings that appear anomalous, valuable, or potentially
    vulnerable. It creates a "lead" for a human operator to pick up.

    Args:
        title: A brief, descriptive summary of the finding.
        interest_level: The priority of the finding. Must be one of:
            - "high": Urgent. Potential for immediate impact or exploitation. (exposed credentials, pre-authentication vulnerability).
            - "medium": Noteworthy. Suggests a potential weakness or area for deeper investigation. (debug endpoint, verbose error messages, PII exposure).
            - "low": Informational. Provides useful context but is not an immediate risk. (software version disclosure, interesting file path).
        justification: A technical, markdown-formatted explanation. Detail *why* the finding is interesting, what its potential impact is, and suggest next steps for a human analyst.
    """
    from dreadnode import log_metric, log_output, tag

    interest_level = interest_level.lower().strip()
    if interest_level not in ["high", "medium", "low"]:
        interest_level = "medium"  # Default to medium if invalid

    logger.success(f"Area of Interest - '{title}' [{interest_level}]:\n{justification}\n---")

    tag(f"interest/{interest_level}")
    log_output("markdown", Markdown(f"# {title} ({interest_level})\n\n{justification}"))
    log_metric("count", 1, mode="count")

    return "Highlighted."
