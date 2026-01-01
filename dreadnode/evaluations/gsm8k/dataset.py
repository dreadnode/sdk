"""GSM8K dataset loading and problem definition."""

import re
import typing as t
from dataclasses import dataclass


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
    """
    Load a sample of GSM8K problems for demonstration.

    Returns a list of dicts with 'question' and 'numeric_answer' keys,
    matching the schema of the actual GSM8K dataset.
    """
    return [
        {
            "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
            "numeric_answer": 18.0,
        },
        {
            "question": "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "numeric_answer": 3.0,
        },
        {
            "question": "Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?",
            "numeric_answer": 70000.0,
        },
        {
            "question": "Every day, Wendi feeds each of her chickens three cups of mixed chicken feed, containing seeds, mealworms and vegetables to help keep them healthy. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?",
            "numeric_answer": 20.0,
        },
        {
            "question": "Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?",
            "numeric_answer": 64.0,
        },
    ]
