from loguru import logger

from dreadnode import log_metric, log_output
from dreadnode.agent.tools.base import tool
from dreadnode.data_types import Markdown


@tool
async def give_up(reason: str) -> None:
    """
    Give up on your task.
    """
    logger.info(f"Agent gave up on the task: {reason}")
    log_output("complete_task_summary", Markdown(f"## Gave up on task\n\n{reason}"))
    log_metric("agent_marked_complete", 1)