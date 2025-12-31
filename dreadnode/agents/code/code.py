from dreadnode.core.agents import Agent
from dreadnode.core.agents.stopping import never
from dreadnode.hooks.summarize import summarize_when_long
from dreadnode.tools.execute import command, python
from dreadnode.tools.fs import LocalFilesystem
from dreadnode.tools.planning import think, update_todo
from dreadnode.tools.tasking import finish_task, give_up_on_task

code_agent = Agent(
    name="code",
    description="An AI coding assistant",
    model="anthropic/claude-sonnet-4-20250514",
    instructions="""
    You are an expert software engineer. You have access to a filesystem
    and can execute shell commands and Python code.

    When editing files:
    - Read before writing
    - Make minimal, targeted changes
    - Verify your changes worked

    When debugging:
    - Form a hypothesis
    - Test it
    - Iterate
    """,
    tools=[
        LocalFilesystem(path=".", variant="write"),
        command,
        python,
        update_todo,
        think,
        finish_task,
        give_up_on_task,
    ],
    stop_conditions=[never()],
    hooks=[summarize_when_long],
    max_steps=50,
)
