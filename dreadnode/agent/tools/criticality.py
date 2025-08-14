from loguru import logger

from dreadnode import log_metric, log_output, tag
from dreadnode.agent.tools.base import tool
from dreadnode.data_types import Markdown


@tool
async def report_finding(file: str, method: str, criticality: str, content: str) -> str:
    """
    Report a finding regarding areas or interest or vulnerabilities.

    for criticality, use:
    - "critical"
    - "high"
    - "medium"
    - "low"
    - "info"
    """

    logger.success(f"Reporting finding for {file} ({method}) [{criticality}]:")

    log_output(
        "finding",
        Markdown(
            f"# Finding\n\nFile: `{file}`\nMethod: `{method}`\nCriticality: `{criticality}`\nContent:\n\n{content}"
        ),
    )
    log_metric("num_reports", 1, mode="count", to="run")
    tag(criticality)
    return "Reported"
