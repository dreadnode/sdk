"""Agent Example - Agents with Tools and Hooks.

This example demonstrates:
- dn.Agent - Create an agent with tools
- @dn.tool - Define tools for the agent
- Hooks - React to agent events
- Stop conditions - When to stop the agent
- Trajectory - Access agent execution history

Run with:
    python examples/agent.py

Note: Requires a configured LLM provider (e.g., GROQ_API_KEY).
"""

import asyncio

import dreadnode as dn
from dreadnode.core.agents.events import GenerationStep
from dreadnode.core.agents.reactions import RetryWithFeedback
from dreadnode.core.agents.stopping import tool_use
from dreadnode.core.hook import hook


# Define tools for the agent
@dn.tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A math expression like "2 + 3 * 4" or "100 / 5".

    Returns:
        The result as a string, or an error message.

    Examples:
        calculate("2 + 2") -> "4"
        calculate("10 * 5") -> "50"
    """
    try:
        # Only allow safe characters
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "Error: Invalid characters in expression"
        result = eval(expression)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@dn.tool
def submit_answer(answer: float) -> str:
    """Submit your final numerical answer.

    Call this tool when you have computed the final answer.

    Args:
        answer: The final numerical answer.

    Returns:
        Confirmation of the submitted answer.
    """
    return f"SUBMITTED: {answer}"


# Define hooks to control agent behavior
@hook(GenerationStep)
async def ensure_reasoning(event: GenerationStep):
    """Ensure the agent shows its work before submitting.

    If the agent tries to submit too early (before step 2),
    ask it to show its calculation steps first.
    """
    if not event.messages:
        return None

    content = str(event.messages[-1].content).lower()

    # If trying to submit answer too early, ask for reasoning
    if "submit_answer" in content and event.step < 2:
        return RetryWithFeedback(
            feedback="Please show your calculation steps first before submitting."
        )
    return None


# Create the agent
agent = dn.Agent(
    name="MathSolver",
    model="groq/llama-3.3-70b-versatile",  # Or use your preferred model
    instructions="""You are a math problem solver.

Your job is to:
1. Read the math problem carefully
2. Use the calculate() tool to perform any arithmetic
3. Show your work step by step
4. Use submit_answer() with the final numerical answer

Always verify your calculations before submitting.""",
    tools=[calculate, submit_answer],
    stop_conditions=[tool_use("submit_answer")],  # Stop when answer is submitted
    hooks=[ensure_reasoning],
    max_steps=10,
)


async def main():
    dn.configure(server="local")

    print("Math Agent Example")
    print("=" * 50)

    # Run the agent within a tracked run
    with dn.run("math-agent-demo"):
        problem = "If John has 5 apples and buys 3 more, then gives 2 to Mary, how many does he have?"

        print(f"\nProblem: {problem}\n")
        print("Agent working...")
        print("-" * 50)

        # Run the agent
        trajectory = await agent.run(problem)

        # Print trajectory summary
        print(f"\nTrajectory Summary:")
        print(f"  Steps: {len(trajectory.steps)}")
        print(f"  Messages: {len(trajectory.messages)}")

        # Print each step
        print(f"\nStep-by-step:")
        for step in trajectory.steps:
            step_type = type(step).__name__
            print(f"  Step {step.step}: {step_type}")

            # Show tool calls if any
            if hasattr(step, "tool_calls") and step.tool_calls:
                for tc in step.tool_calls:
                    print(f"    -> Tool: {tc.name}({tc.arguments})")

        # Print token usage if available
        if trajectory.usage:
            print(f"\nToken usage:")
            print(f"  Input: {trajectory.usage.input_tokens}")
            print(f"  Output: {trajectory.usage.output_tokens}")
            print(f"  Total: {trajectory.usage.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
