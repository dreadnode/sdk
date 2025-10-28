import typing as t
from collections import Counter

from loguru import logger
from pydantic import BaseModel, Field

from dreadnode.agent.tools.base import tool


class TodoItem(BaseModel):
    """Represents a single task in the todo list."""

    id: str = Field(
        ..., description="A unique identifier for the todo item (e.g., a UUID or a simple number)."
    )
    content: str = Field(..., min_length=1, description="The descriptive content of the task.")
    status: t.Literal["pending", "in_progress", "completed"] = Field(
        ..., description="The current status of the task."
    )
    priority: t.Literal["high", "medium", "low"] = Field(
        ..., description="The priority level of the task."
    )


@tool(catch=True)
def update_todo(todos: t.Annotated[list[TodoItem], "The full, updated list of todo items."]) -> str:
    """
    Use this tool to create and manage a structured task list for your current session.
    This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.
    It also helps the user understand the progress of the task and overall progress of their requests.

    ## When to Use This Tool

    1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
    2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
    3. User explicitly requests todo list - When the user directly asks you to use the todo list
    4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
    5. After receiving new instructions - Immediately capture user requirements as todos
    6. When you start working on a task - Mark it as in_progress BEFORE beginning work. Ideally you should only have one todo as in_progress at a time
    7. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation

    ## When NOT to Use This Tool

    1. There is only a single, straightforward task
    2. The task is trivial and tracking it provides no organizational benefit
    3. The task can be completed in less than 3 trivial steps
    4. The task is purely conversational or informational

    NOTE that you should not use this tool if there is only one trivial task to do. In this case you are better off just doing the task directly.

    ## Task States and Management

    1. **Task States**: Use these states to track progress:
       - pending: Task not yet started
       - in_progress: Currently working on (limit to ONE task at a time)
       - completed: Task finished successfully

    2. **Task Management**:
       - Update task status in real-time as you work
       - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
       - Only have ONE task in_progress at any time
       - Complete current tasks before starting new ones
       - Remove tasks that are no longer relevant from the list entirely

    3. **Task Completion Requirements**:
       - ONLY mark a task as completed when you have FULLY accomplished it
       - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
       - When blocked, create a new task describing what needs to be resolved
       - Never mark a task as completed if:
         - Tests are failing
         - Implementation is partial
         - You encountered unresolved errors
         - You couldn't find necessary files or dependencies

    4. **Task Breakdown**:
       - Create specific, actionable items
       - Break complex tasks into smaller, manageable steps
       - Use clear, descriptive task names

    When in doubt, use this tool. Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully.
    """
    from dreadnode import log_metric, log_output

    status_counts = Counter(t.status for t in todos)

    log_metric("num_todos", len(todos))
    log_metric("completed_todos", status_counts["completed"])
    log_metric("in_progress_todos", status_counts["in_progress"])
    log_metric("pending_todos", status_counts["pending"])

    log_output("todos", todos)

    if not todos:
        logger.info("Todo list cleared.")
        return "Todo list cleared."

    status_log = f"Updated todo list with {len(todos)} tasks:\n"
    for todo in todos:
        status = (
            "✅" if todo.status == "completed" else ("⏳" if todo.status == "in_progress" else "📌")
        )
        status_log += f"{status} {todo.content} (priority: {todo.priority})\n"

    logger.info(status_log)

    return (
        f"Updated todo list with {len(todos)} tasks. "
        f"{status_counts['completed']} completed, "
        f"{status_counts['in_progress']} in progress, "
        f"{status_counts['pending']} pending."
    )


@tool
def think(thought: str) -> None:
    """
    Records a thought, reflection, or plan to document your reasoning process.

    This tool acts as your internal monologue, allowing you to articulate your strategy. Use it to:
    - Break down a complex problem into smaller steps.
    - Formulate a multi-step plan before you act.
    - Interpret the results of another tool's output.
    - Document a change in strategy (self-correction).

    A clear chain of thought is essential for explaining your actions.

    ## Best Practices
    - Do Not Substitute for Action**: After thinking, you must call the appropriate \
    tool to execute your plan. This tool performs no action on its own.
    - Do Not Repeat Information**: Never use this to repeat the output of other tools. \
    Use it to state your *conclusion* or *next step* based on that output.

    Args:
        thought: A clear, concise statement of your thought process or plan.
    """
    logger.info(f"Agent thought: {thought}")
