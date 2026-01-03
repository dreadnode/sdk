"""Shared utilities for SDK examples.

This module provides common functionality used across examples to reduce
code duplication and demonstrate best practices.
"""

import os
import re


def setup_training_env() -> None:
    """Set required environment variables for training.

    VLLM requires spawn method for multiprocessing to avoid CUDA fork issues.
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def extract_gsm8k_answer(text: str) -> float | None:
    """Extract numerical answer from GSM8K format.

    GSM8K answers are formatted as "#### <number>" at the end.

    Args:
        text: Text containing the answer in GSM8K format.

    Returns:
        The extracted numerical answer, or None if not found.

    Examples:
        >>> extract_gsm8k_answer("The total is #### 42")
        42.0
        >>> extract_gsm8k_answer("Therefore, #### 1,234.56")
        1234.56
    """
    # Match #### followed by optional whitespace and a number (with optional commas/decimals)
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


def normalize_answer(answer: str) -> str | None:
    """Normalize a numerical answer string.

    Args:
        answer: Answer string to normalize.

    Returns:
        Normalized answer string, or None if not a valid number.

    Examples:
        >>> normalize_answer("42")
        "42"
        >>> normalize_answer("1,234.56")
        "1234.56"
    """
    try:
        # Remove commas and whitespace
        cleaned = answer.replace(",", "").strip()
        # Try to parse as float and format consistently
        value = float(cleaned)
        # Return as integer string if whole number, else float
        if value == int(value):
            return str(int(value))
        return str(value)
    except ValueError:
        return None


def load_gsm8k(split: str = "train", max_samples: int = 100) -> list[dict]:
    """Load GSM8K dataset samples.

    Args:
        split: Dataset split to load ("train" or "test").
        max_samples: Maximum number of samples to load.

    Returns:
        List of dicts with "question" and "answer" keys.

    Example:
        >>> samples = load_gsm8k(split="train", max_samples=10)
        >>> samples[0]
        {"question": "Janet's ducks...", "answer": 50.0}
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)

    items = []
    for i, item in enumerate(ds):
        if i >= max_samples:
            break
        items.append({
            "question": item["question"],
            "answer": extract_gsm8k_answer(item["answer"]),
        })
    return items


def format_math_prompt(question: str) -> str:
    """Format a math question as a prompt.

    Args:
        question: The math question to format.

    Returns:
        Formatted prompt string.
    """
    return f"Question: {question}\n\nAnswer:"


def create_math_reward_fn(
    prompts: list[str],
    answers: list[float | None],
    tolerance: float = 0.01,
) -> callable:
    """Create a reward function for math problems.

    Args:
        prompts: List of prompt strings.
        answers: List of expected answers (parallel to prompts).
        tolerance: Numerical tolerance for answer comparison.

    Returns:
        Reward function that takes (prompts, completions) and returns rewards.

    Example:
        >>> prompts = ["What is 2+2?", "What is 3*4?"]
        >>> answers = [4.0, 12.0]
        >>> reward_fn = create_math_reward_fn(prompts, answers)
        >>> reward_fn(prompts, ["The answer is 4", "The answer is 12"])
        [1.0, 1.0]
    """
    prompt_to_answer = dict(zip(prompts, answers))

    def reward_fn(prompts: list[str], completions: list[str]) -> list[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            expected = prompt_to_answer.get(prompt)
            if expected is None:
                rewards.append(0.0)
                continue

            extracted = extract_gsm8k_answer(completion)
            if extracted is not None and abs(extracted - expected) < tolerance:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        return rewards

    return reward_fn
