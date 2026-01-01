"""GSM8K math problem solving agent."""

from dreadnode.agents import Agent
from dreadnode.core.agents.events import GenerationStep, ToolEnd
from dreadnode.core.agents.reactions import Reaction, RetryWithFeedback
from dreadnode.core.agents.stopping import tool_use
from dreadnode.core.conditions import tool_called
from dreadnode.core.hook import hook
from dreadnode.evaluations.gsm8k.tools import calculate, submit_answer


# =============================================================================
# Hook Definitions
# =============================================================================


@hook(GenerationStep)
async def check_reasoning_quality(event: GenerationStep) -> Reaction | None:
    """Check if the agent is showing good reasoning patterns."""
    last_message = event.messages[-1] if event.messages else None
    if not last_message:
        return None

    content = str(last_message.content).lower()

    # Encourage showing work before submitting early
    if "submit_answer" in content and event.step < 3:
        return RetryWithFeedback(
            feedback="Please show your work step by step using the calculate tool before submitting."
        )

    return None


@hook(ToolEnd, condition=tool_called("calculate"))
async def validate_calculation(event: ToolEnd) -> Reaction | None:
    """Validate calculation results and provide feedback on errors."""
    result = str(event.result)

    if "Error" in result:
        return RetryWithFeedback(
            feedback=f"Your calculation had an error: {result}. Please reformulate."
        )

    return None


# =============================================================================
# Instruction Variants for Optimization
# =============================================================================

INSTRUCTION_VARIANTS = {
    "concise": """
    Solve math word problems step by step.
    Use 'calculate' for arithmetic. Use 'submit_answer' for the final numeric answer.
    """,
    "detailed": """
    You are a careful math problem solver. Follow these steps:

    1. READ: Understand what the problem is asking.
    2. IDENTIFY: List the numbers and what they represent.
    3. PLAN: Decide what operations to perform.
    4. CALCULATE: Use the calculate tool for each step.
    5. VERIFY: Double-check your work.
    6. SUBMIT: Use submit_answer with the final number.
    """,
    "structured": """
    Solve the problem using this format:

    GIVEN: [List known values]
    FIND: [What we need to calculate]
    STEPS: [Use calculate tool for each step]
    ANSWER: [Use submit_answer tool]
    """,
}

DEFAULT_INSTRUCTIONS = """
You are a math problem-solving agent. Your goal is to solve grade-school math word problems.

APPROACH:
1. Read the problem carefully and identify the key quantities and relationships.
2. Break down the problem into smaller steps.
3. Use the 'calculate' tool to perform each arithmetic operation.
4. Show your reasoning by explaining each step.
5. When you have the final answer, use 'submit_answer' to provide the numeric result.

IMPORTANT:
- Always show your work using the calculate tool.
- Double-check your calculations.
- The final answer should be a single number.
"""


def create_math_agent(
    model: str = "groq/moonshotai/kimi-k2-instruct-0905",
    max_steps: int = 15,
    instructions: str | None = None,
    use_hooks: bool = True,
) -> Agent:
    """
    Create a math problem-solving agent for GSM8K.

    Args:
        model: The LLM model to use.
        max_steps: Maximum number of steps before stopping.
        instructions: Custom instructions for the agent.
        use_hooks: Whether to use reasoning hooks.

    Returns:
        Configured Agent instance.
    """
    hooks = [check_reasoning_quality, validate_calculation] if use_hooks else []

    return Agent(
        name="GSM8K Math Agent",
        description="An agent that solves grade-school math word problems step by step.",
        model=model,
        instructions=instructions or DEFAULT_INSTRUCTIONS,
        max_steps=max_steps,
        tools=[calculate, submit_answer],
        stop_conditions=[tool_use("submit_answer")],
        hooks=hooks,
        tags=["math", "gsm8k", "reasoning"],
    )


# Create a default agent instance for discovery
MathAgent = create_math_agent()
