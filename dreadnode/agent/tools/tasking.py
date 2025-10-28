from loguru import logger

from dreadnode.agent.reactions import Fail, Finish
from dreadnode.agent.tools.base import tool


@tool
async def finish_task(success: bool, summary: str) -> None:  # noqa: ARG001, FBT001
    """
    Concludes the task by reporting a final status and a comprehensive summary.

    This is the **final tool** to call when your planned sequence of actions is complete, \
    regardless of whether the outcome was successful. Use it when you have no more \
    steps to take and are ready to present a final report.

    ## Best Practices
    - Honest Status: The `success` flag must accurately reflect the final outcome. \
    If any part of the task failed or objectives were not met, it must be `False`.
    - Comprehensive Summary: The `summary` is your final report. It must be a complete, \
    markdown-formatted document detailing all actions taken, tools used, and the results.

    Args:
        success: True if the task's objectives were fully met, False otherwise.
        summary: A complete markdown-formatted report of all actions and outcomes.
    """
    from dreadnode import log_metric

    log_func = logger.success if success else logger.warning
    log_func(f"Agent finished the task (success={success})")
    log_metric("task_success", success)

    raise Finish if success else Fail("Agent marked the task as failed.")


@tool
async def give_up_on_task(reason: str) -> None:
    """
    Aborts the task when you are irrecoverably stuck and cannot make progress.

    This tool is a last resort and should only be used when you have exhausted all \
    possible strategies and alternative approaches. It signals that you were unable \
    to complete your assigned process.

    ## Best Practices
    - Do Not Use for a Failed Outcome**: If the `finish_task` tool is available, use it to report failures. \
    This tool is strictly for when you cannot *finish* your work.
    - Provide a Clear Justification**: The `reason` must clearly explain why you are stuck. \
    Detail the final obstacle you could not overcome and the approaches you already tried.

    Args:
        reason: A concise explanation of why you are unable to continue the task.
    """
    from dreadnode import log_metric

    logger.warning(f"Agent gave up on the task: {reason}")
    log_metric("task_give_up", 1)

    raise Fail("Agent gave up on the task.")
