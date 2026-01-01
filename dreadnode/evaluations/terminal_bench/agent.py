"""Terminal agent for Terminal-Bench evaluation.

Following the Terminal-Bench design, the agent has access to a single tool:
a bash session where it can execute any command. This mimics how Terminus
operates - by sending commands into an interactive terminal session.
"""

from dreadnode import tool
from dreadnode.agents import Agent
from dreadnode.environments.docker import DockerEnvironment

DEFAULT_INSTRUCTIONS = """
You are an expert terminal operator. Your goal is to complete tasks in a Linux environment.

You have a single tool: `bash` - which executes commands in a terminal session.

APPROACH:
1. Understand what the task requires
2. Explore if needed (ls, cat, pwd, etc.)
3. Execute commands to accomplish the goal
4. Verify your work succeeded

TIPS:
- Use standard Unix commands (cat, echo, sed, grep, mkdir, etc.)
- Create files with: echo 'content' > file.txt or cat << 'EOF' > file.txt
- Edit files with sed, or rewrite with cat/echo
- Install packages with pip or apt if needed
- Check exit codes and output for errors
- Use absolute paths when helpful

When the task is complete, stop issuing commands.
"""


def create_bash_tool(environment: DockerEnvironment):
    """Create a single bash tool bound to the Docker environment."""

    @tool
    async def bash(command: str) -> str:
        """
        Execute a bash command in the terminal.

        Args:
            command: The bash command to execute.

        Returns:
            The command output (stdout and stderr combined).
        """
        exit_code, output = await environment.container.run(
            command,
            timeout=60,
            stream_output=False,
        )

        # Include exit code in output if non-zero
        if exit_code != 0:
            return f"{output}\n[Exit code: {exit_code}]"
        return output if output else "[No output]"

    return bash


def create_terminal_agent(
    environment: DockerEnvironment,
    model: str = "groq/moonshotai/kimi-k2-instruct-0905",
    max_steps: int = 30,
    instructions: str | None = None,
) -> Agent:
    """
    Create a terminal agent with a single bash tool.

    Following Terminal-Bench's Terminus design, the agent has only one tool:
    a bash command executor. This forces the agent to use standard Unix
    commands for all operations.

    Args:
        environment: The Docker environment (must be set up before use).
        model: The LLM model to use.
        max_steps: Maximum number of steps before stopping.
        instructions: Custom instructions for the agent.

    Returns:
        Configured Agent instance.
    """
    bash_tool = create_bash_tool(environment)

    return Agent(
        name="Terminal Agent",
        description="An agent that completes tasks using bash commands.",
        model=model,
        instructions=instructions or DEFAULT_INSTRUCTIONS,
        max_steps=max_steps,
        tools=[bash_tool],
        tags=["terminal", "docker", "bash"],
    )
