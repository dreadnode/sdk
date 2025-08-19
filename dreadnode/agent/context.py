import sys
from contextvars import ContextVar

if sys.version_info >= (3, 11):
    from asyncio import TaskGroup
else:
    from taskgroup import TaskGroup

current_task_group: ContextVar[TaskGroup | None] = ContextVar("current_task_group", default=None)
