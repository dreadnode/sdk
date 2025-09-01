from loguru import logger

from dreadnode import log_metric, log_output
from dreadnode.agent.reactions import Fail, Finish
from dreadnode.agent.tools.base import tool
from dreadnode.data_types import Markdown


@tool
async def complete_successfully(summary: str) -> None:
    """
    Mark your task as successfully completed with a markdown summary of actions taken.

    ## When to Use This Tool
    This tool should be called under the following circumstances:
    1.  **All TODOs are complete**: If you are managing todos, every task in your TODO list has been marked as 'completed'.
    2.  **No more actions**: You have no further actions to take and have addressed all aspects of the user's request.
    4.  **Final Summary**: You are ready to provide a comprehensive summary of all actions taken.

    ## When NOT to Use This Tool
    Do not use this tool if:
    1.  **The task failed**: Use `mark_as_failed` instead.
    2.  **You are in the middle of a multi-step process**: The overall task is not yet finished.
    3.  **A recoverable error has occurred**: You should first attempt to fix the error through all available means.
    4.  **You are waiting for user feedback**: The task is paused, not finished.

    ## Best Practices
    *   **Final Step**: This should be the absolute last tool you call. Once invoked, your task is considered finished.
    *   **Comprehensive Summary**: The `summary` should be a complete and detailed markdown-formatted report of everything you did, including steps taken, tools used, and the final outcome. This is your final report to the user.
    """
    logger.success("Agent finished the task successfully:")
    logger.info(summary)
    logger.info("---")

    log_metric("task_success", True)
    log_output("task_summary", Markdown(summary))

    raise Finish


@tool
async def mark_as_failed(summary: str) -> None:
    """
    Mark your task as failed with a markdown summary of actions taken and reasons for failure.

    ## When to Use This Tool
    This tool should be called under the following circumstances:
    1.  **Irrecoverable failure**: You have encountered an error that you cannot resolve, and there are no further steps you can take.
    2.  **Final Summary**: You are ready to provide a comprehensive summary of what failed and why.

    ## When NOT to Use This Tool
    Do not use this tool if:
    1.  **The task succeeded**: Use `complete_successfully` instead.
    2.  **You are in the middle of a multi-step process**: The overall task is not yet finished.
    3.  **A recoverable error has occurred**: You should first attempt to fix the error through all available means.
    4.  **You are waiting for user feedback**: The task is paused, not finished.

    ## Best Practices
    *   **Final Step**: This should be the absolute last tool you call. Once invoked, your task is considered finished.
    *   **Comprehensive Summary**: The `summary` should be a complete and detailed markdown-formatted report of what you attempted, where the process failed, and why. This is your final report to the user.
    """
    logger.warning("Agent finished the task with failure:")
    logger.info(summary)
    logger.info("---")

    log_metric("task_success", False)
    log_output("task_summary", Markdown(summary))

    raise Fail("Agent marked the task as failed.")
