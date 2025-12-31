"""
GSM8K Agent Evaluation and Optimization Example

This module demonstrates a complete end-to-end workflow using the Agent's
built-in evaluate() and optimize() methods:

1. Defining an agent that solves GSM8K math problems
2. Creating scorers to evaluate solution quality
3. Using agent.evaluate() for single goal and dataset evaluation
4. Using agent.optimize() for hyperparameter search
5. Using hooks to provide feedback and improve agent behavior

The GSM8K (Grade School Math 8K) benchmark tests mathematical reasoning
with grade-school level word problems requiring multi-step solutions.
"""

import asyncio
import re
import typing as t
from dataclasses import dataclass

import dreadnode as dn
from dreadnode.core.agents import Agent
from dreadnode.core.agents.events import (
    AgentStep,
    GenerationStep,
    ToolEnd,
)
from dreadnode.core.agents.reactions import Reaction, RetryWithFeedback
from dreadnode.core.agents.trajectory import Trajectory
from dreadnode.core.conditions import tool_called
from dreadnode.core.hook import hook
from dreadnode.core.judge import Judge, Rubric
from dreadnode.core.metric import Metric
from dreadnode.core.scorer import scorer, weighted_avg
from dreadnode.core.stopping import stop_condition

# =============================================================================
# GSM8K Dataset Definition
# =============================================================================


@dataclass
class GSM8KProblem:
    """A single GSM8K problem with question and expected answer."""

    question: str
    answer: str
    numeric_answer: float

    @classmethod
    def from_dict(cls, data: dict[str, t.Any]) -> "GSM8KProblem":
        """Parse a GSM8K problem from a dictionary."""
        answer = data.get("answer", "")
        numeric_match = re.search(r"####\s*([\d,.-]+)", answer)
        numeric_answer = float(numeric_match.group(1).replace(",", "")) if numeric_match else 0.0

        return cls(
            question=data.get("question", ""),
            answer=answer,
            numeric_answer=numeric_answer,
        )


def load_gsm8k_sample() -> list[dict[str, t.Any]]:
    """Load a sample of GSM8K problems for demonstration."""
    return [
        {
            "goal": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "numeric_answer": 18.0,
        },
        {
            "goal": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "numeric_answer": 3.0,
        },
        {
            "goal": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            "numeric_answer": 70000.0,
        },
        {
            "goal": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
            "numeric_answer": 20.0,
        },
        {
            "goal": "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?",
            "numeric_answer": 64.0,
        },
    ]


# =============================================================================
# Tool Definitions
# =============================================================================


@dn.tool
def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.

    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 3 * 4")

    Returns:
        The result of the calculation as a string.
    """
    try:
        allowed_chars = set("0123456789+-*/().% ")
        if not all(c in allowed_chars for c in expression):
            return "Error: Expression contains invalid characters."

        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e!s}"


@dn.tool(stop=True)
def submit_answer(answer: float) -> str:
    """
    Submit the final numeric answer to the problem.

    This tool stops the agent when called.

    Args:
        answer: The final numeric answer to submit.

    Returns:
        Confirmation of the submitted answer.
    """
    return f"Answer submitted: {answer}"


# =============================================================================
# Scorer Definitions
# =============================================================================


def extract_submitted_answer(trajectory: Trajectory) -> float | None:
    """Extract the submitted answer from the agent's trajectory."""
    for step in reversed(trajectory.steps):
        if isinstance(step, ToolEnd) and step.tool_call.name == "submit_answer":
            try:
                args = step.tool_call.arguments
                if isinstance(args, dict) and "answer" in args:
                    return float(args["answer"])
            except (ValueError, TypeError):
                pass
    return None


@scorer(name="answer_correct")
async def answer_correct_scorer(
    trajectory: Trajectory,
    expected_answer: float = dn.DatasetField("numeric_answer"),
) -> float:
    """
    Score whether the agent's submitted answer matches the expected answer.

    Uses dn.DatasetField to automatically pull the expected answer from the dataset.
    Returns 1.0 for correct, 0.0 for incorrect or no answer.
    """
    submitted = extract_submitted_answer(trajectory)
    if submitted is None:
        return 0.0

    tolerance = abs(expected_answer) * 0.001 if expected_answer != 0 else 0.001
    return 1.0 if abs(submitted - expected_answer) <= tolerance else 0.0


@scorer(name="reasoning_quality")
async def reasoning_quality_scorer(trajectory: Trajectory) -> float:
    """
    Score the quality of the agent's reasoning process.
    """
    calculation_count = 0
    has_answer = False

    for step in trajectory.steps:
        if isinstance(step, ToolEnd):
            if step.tool_call.name == "calculate":
                calculation_count += 1
            elif step.tool_call.name == "submit_answer":
                has_answer = True

    calc_score = min(1.0, calculation_count / 5.0)
    answer_score = 1.0 if has_answer else 0.0

    return calc_score * 0.5 + answer_score * 0.5


@scorer(name="efficiency")
async def efficiency_scorer(trajectory: Trajectory) -> float:
    """
    Score the efficiency of the agent's problem-solving.
    """
    step_count = len(trajectory.steps)

    if step_count < 3:
        return 0.5
    if step_count <= 8:
        return 1.0
    if step_count <= 15:
        return 0.7
    return max(0.2, 1.0 - (step_count - 15) * 0.05)


# Composite scorer using library's built-in weighted_avg
# @scorer decorated functions are already Scorer instances
gsm8k_composite_scorer = weighted_avg(
    (answer_correct_scorer, 0.6),
    (reasoning_quality_scorer, 0.3),
    (efficiency_scorer, 0.1),
    name="gsm8k_composite",
)


# =============================================================================
# Judge Implementation
# =============================================================================


@dataclass
class GSM8KRubric(Rubric):
    """Rubric for evaluating GSM8K problem solutions."""

    name: str = "gsm8k_quality"
    expected_answer: float = 0.0
    weight_correctness: float = 0.6
    weight_reasoning: float = 0.3
    weight_efficiency: float = 0.1


class GSM8KJudge(Judge[GSM8KRubric]):
    """Judge for evaluating GSM8K agent performance."""

    def evaluate(self, result: Trajectory) -> Metric:
        """Evaluate the agent's trajectory against the rubric."""
        submitted = extract_submitted_answer(result)
        if submitted is None:
            correctness = 0.0
        else:
            tolerance = (
                abs(self.rubric.expected_answer) * 0.001
                if self.rubric.expected_answer != 0
                else 0.001
            )
            correctness = 1.0 if abs(submitted - self.rubric.expected_answer) <= tolerance else 0.0

        calc_count = sum(
            1
            for step in result.steps
            if isinstance(step, ToolEnd) and step.tool_call.name == "calculate"
        )
        reasoning = min(1.0, calc_count / 5.0)

        step_count = len(result.steps)
        if 3 <= step_count <= 8:
            efficiency = 1.0
        elif step_count < 3:
            efficiency = 0.5
        else:
            efficiency = max(0.2, 1.0 - (step_count - 8) * 0.1)

        final_score = (
            correctness * self.rubric.weight_correctness
            + reasoning * self.rubric.weight_reasoning
            + efficiency * self.rubric.weight_efficiency
        )

        return Metric(
            value=final_score,
            attributes={
                "correctness": correctness,
                "reasoning": reasoning,
                "efficiency": efficiency,
            },
        )


# =============================================================================
# Stop Conditions
# =============================================================================


@stop_condition(name="answer_submitted")
def answer_submitted(steps: list[AgentStep]) -> bool:
    """Stop when an answer has been submitted."""
    for step in steps:
        if isinstance(step, ToolEnd) and step.tool_call.name == "submit_answer":
            return True
    return False


@stop_condition(name="max_calculations")
def max_calculations_condition(steps: list[AgentStep], limit: int = 10) -> bool:
    """Stop after a maximum number of calculations."""
    calc_count = sum(
        1 for step in steps if isinstance(step, ToolEnd) and step.tool_call.name == "calculate"
    )
    return calc_count >= limit


# =============================================================================
# Hook Definitions
# =============================================================================


class MathReasoningHooks:
    """Hooks to improve agent behavior during math problem solving."""

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self.retry_count = 0

    @hook(GenerationStep)
    async def check_reasoning_quality(self, event: GenerationStep) -> Reaction | None:
        """
        Check if the agent is showing good reasoning patterns.
        """
        last_message = event.messages[-1] if event.messages else None
        if not last_message:
            return None

        content = str(last_message.content).lower()

        if "submit_answer" in content and event.step < 3:
            if self.retry_count < self.max_retries:
                self.retry_count += 1
                return RetryWithFeedback(
                    feedback="Please show your work step by step using the calculate tool before submitting."
                )

        return None

    @hook(ToolEnd, condition=tool_called("calculate"))
    async def validate_calculation(self, event: ToolEnd) -> Reaction | None:
        """Validate calculation results and provide feedback on errors."""
        result = str(event.result)

        if "Error" in result:
            return RetryWithFeedback(
                feedback=f"Your calculation had an error: {result}. Please reformulate."
            )

        return None


class AdaptiveMathHooks:
    """Advanced hooks that adapt based on agent performance."""

    @hook(GenerationStep)
    async def adaptive_guidance(self, event: GenerationStep) -> Reaction | None:
        """Provide adaptive guidance based on the agent's progress."""
        step = event.step
        messages = event.messages

        if step >= 10:
            has_answer = any("submit_answer" in str(m.content).lower() for m in messages[-3:])
            if not has_answer:
                return RetryWithFeedback(
                    feedback="You've done extensive calculations. Please submit your final answer."
                )

        return None


# =============================================================================
# Agent Definition
# =============================================================================

# Instruction variants for optimization
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


def create_math_agent(
    model: str = "anthropic/claude-sonnet-4-20250514",
    max_steps: int = 15,
    instructions: str | None = None,
    use_advanced_hooks: bool = False,
) -> Agent:
    """
    Create a math problem-solving agent for GSM8K.

    Args:
        model: The LLM model to use.
        max_steps: Maximum number of steps before stopping.
        instructions: Custom instructions for the agent.
        use_advanced_hooks: Whether to use advanced adaptive hooks.

    Returns:
        Configured Agent instance.
    """
    default_instructions = """
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

    if use_advanced_hooks:
        hooks_instance = AdaptiveMathHooks()
        hooks = [hooks_instance.adaptive_guidance]
    else:
        hooks_instance = MathReasoningHooks()
        hooks = [
            hooks_instance.check_reasoning_quality,
            hooks_instance.validate_calculation,
        ]

    return Agent(
        name="GSM8K Math Agent",
        description="An agent that solves grade-school math word problems step by step.",
        model=model,
        instructions=instructions or default_instructions,
        max_steps=max_steps,
        tools=[calculate, submit_answer],  # Use decorated tools directly
        stop_conditions=[answer_submitted],
        hooks=hooks,
        tags=["math", "gsm8k", "reasoning"],
    )


# =============================================================================
# Main Pipeline Using Agent's Built-in Methods
# =============================================================================


async def main():
    """
    Main function demonstrating the complete evaluation and optimization pipeline
    using the Agent's built-in evaluate() and optimize() methods.
    """
    print("=" * 70)
    print("GSM8K Agent Evaluation and Optimization Pipeline")
    print("Using Agent.evaluate() and Agent.optimize() methods")
    print("=" * 70)

    # Load dataset
    print("\n[1] Loading GSM8K sample dataset...")
    dataset = load_gsm8k_sample()
    print(f"    Loaded {len(dataset)} problems")

    # Create base agent
    print("\n[2] Creating base math agent...")
    base_agent = create_math_agent()
    print(f"    Agent: {base_agent.name}")
    print(f"    Max steps: {base_agent.max_steps}")
    print(f"    Tools: {[t.name for t in base_agent.all_tools]}")

    # ==========================================================================
    # Single Goal Evaluation using agent.evaluate()
    # ==========================================================================
    print("\n[3] Single goal evaluation using agent.evaluate()...")

    single_result = await base_agent.evaluate(
        goal="Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes with four. She sells the rest for $2 each. How much does she make?",
        scorers=[answer_correct_scorer, reasoning_quality_scorer, efficiency_scorer],
        max_attempts=3,  # Retry up to 3 times
        stop_on_success=True,  # Stop early if correct
    )

    print(f"    Attempts made: {len(single_result.samples)}")
    if single_result.samples:
        best_sample = max(single_result.samples, key=lambda s: s.score)
        print(f"    Best score: {best_sample.score:.2f}")
        print(f"    Passed: {'✓' if best_sample.passed else '✗'}")

    # ==========================================================================
    # Dataset Evaluation using agent.evaluate()
    # ==========================================================================
    print("\n[4] Dataset evaluation using agent.evaluate()...")

    eval_result = await base_agent.evaluate(
        dataset=dataset,
        scorers=[answer_correct_scorer, reasoning_quality_scorer, efficiency_scorer],
        concurrency=2,
        iterations=1,
        max_consecutive_errors=5,
        name="GSM8K Baseline Evaluation",
    )

    # Analyze results
    samples = eval_result.samples
    correct_count = sum(1 for s in samples if s.scores.get("answer_correct", 0) > 0)
    total = len(samples)

    print("    Results:")
    print(f"    - Accuracy: {correct_count}/{total} ({100 * correct_count / total:.1f}%)")
    print(f"    - Pass rate: {eval_result.pass_rate:.1%}")

    # ==========================================================================
    # Optimization using agent.optimize()
    # ==========================================================================
    print("\n[5] Optimization using agent.optimize()...")
    print("    Search space:")
    print("    - max_steps: [8, 12, 15]")
    print("    - instructions: [concise, detailed, structured]")

    # Define search space for optimization
    search_space = {
        "max_steps": [8, 12, 15],
        "instructions": list(INSTRUCTION_VARIANTS.values()),
    }

    best_agent, study_result = await base_agent.optimize(
        dataset=dataset,
        search=search_space,
        search_strategy="grid",  # or "random", "optuna"
        scorers=[answer_correct_scorer],
        directions=["maximize"],
        max_trials=9,  # 3 x 3 grid
        concurrency=2,
        name="GSM8K Optimization Study",
    )

    print(f"    Trials completed: {len(study_result.trials)}")
    if study_result.best_trial:
        print(f"    Best trial score: {study_result.best_trial.score:.2f}")
        print(f"    Best config: max_steps={study_result.best_trial.candidate.get('max_steps')}")

    # ==========================================================================
    # Evaluate the optimized agent
    # ==========================================================================
    print("\n[6] Evaluating optimized agent...")

    optimized_result = await best_agent.evaluate(
        dataset=dataset,
        scorers=[answer_correct_scorer, reasoning_quality_scorer],
        concurrency=2,
    )

    optimized_correct = sum(
        1 for s in optimized_result.samples if s.scores.get("answer_correct", 0) > 0
    )

    print(
        f"    Optimized accuracy: {optimized_correct}/{total} ({100 * optimized_correct / total:.1f}%)"
    )
    print(f"    Improvement: {optimized_correct - correct_count:+d} problems")

    # ==========================================================================
    # Create improved agent with learnings
    # ==========================================================================
    print("\n[7] Creating improved agent with learnings from failures...")

    # Extract learnings from failed samples
    learnings = []
    for sample in eval_result.samples:
        if sample.scores.get("answer_correct", 0) == 0:
            if sample.scores.get("reasoning_quality", 0) < 0.3:
                learnings.append("Always use calculate tool for intermediate steps")

    learnings.extend(
        [
            "For percentage problems, calculate the base amount first",
            "Double-check subtraction operations",
            "Verify the answer makes sense in context",
        ]
    )

    improved_instructions = INSTRUCTION_VARIANTS["detailed"] + "\n\nLEARNED PATTERNS:\n"
    for i, learning in enumerate(learnings[:3], 1):
        improved_instructions += f"{i}. {learning}\n"

    improved_agent = base_agent.with_(
        instructions=improved_instructions,
        max_steps=12,
    )

    print(f"    Applied {len(learnings)} learned patterns")

    # ==========================================================================
    # Final comparison
    # ==========================================================================
    print("\n[8] Final comparison on single problem...")

    test_goal = dataset[0]["goal"]
    expected = dataset[0]["numeric_answer"]

    # Run both agents
    base_trajectory = await base_agent.run(test_goal)
    improved_trajectory = await improved_agent.run(test_goal)

    base_answer = extract_submitted_answer(base_trajectory)
    improved_answer = extract_submitted_answer(improved_trajectory)

    print(f"    Expected: {expected}")
    print(f"    Base agent: {base_answer} ({'✓' if base_answer == expected else '✗'})")
    print(f"    Improved agent: {improved_answer} ({'✓' if improved_answer == expected else '✗'})")

    print("\n" + "=" * 70)
    print("Pipeline complete!")
    print("=" * 70)

    return {
        "base_agent": base_agent,
        "best_agent": best_agent,
        "improved_agent": improved_agent,
        "eval_result": eval_result,
        "study_result": study_result,
    }


async def quick_example():
    """
    A simpler example demonstrating the key agent methods.
    """
    print("Quick GSM8K Agent Example")
    print("-" * 40)

    # Create agent
    agent = create_math_agent(max_steps=10)

    # Simple single-goal evaluation with retries
    result = await agent.evaluate(
        goal="If John has 5 apples and buys 3 more, then gives away 2, how many does he have?",
        scorers=[answer_correct_scorer],
        max_attempts=3,
        stop_on_success=True,
    )

    print(f"Attempts: {len(result.samples)}")
    if result.samples:
        trajectory = result.samples[-1].output
        answer = extract_submitted_answer(trajectory)
        print(f"Answer: {answer}")
        print("Expected: 6")
        print(f"Correct: {'✓' if answer == 6.0 else '✗'}")

    return result


if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(quick_example())
