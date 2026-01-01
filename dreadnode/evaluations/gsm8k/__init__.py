"""
GSM8K Evaluation Module

This module provides a complete evaluation for the GSM8K (Grade School Math 8K)
benchmark, including tools, scorers, agent, and evaluation definitions.
"""

from dreadnode.evaluations.gsm8k.agent import MathAgent, create_math_agent
from dreadnode.evaluations.gsm8k.dataset import GSM8KProblem, load_gsm8k_sample
from dreadnode.evaluations.gsm8k.evaluation import GSM8KEvaluation, create_gsm8k_evaluation
from dreadnode.evaluations.gsm8k.scorers import (
    answer_correct_scorer,
    efficiency_scorer,
    extract_submitted_answer,
    gsm8k_composite_scorer,
    reasoning_quality_scorer,
)
from dreadnode.evaluations.gsm8k.tools import calculate, submit_answer

__all__ = [
    # Dataset
    "GSM8KProblem",
    "load_gsm8k_sample",
    # Tools
    "calculate",
    "submit_answer",
    # Scorers
    "answer_correct_scorer",
    "reasoning_quality_scorer",
    "efficiency_scorer",
    "gsm8k_composite_scorer",
    "extract_submitted_answer",
    # Agent
    "create_math_agent",
    "MathAgent",
    # Evaluation
    "GSM8KEvaluation",
    "create_gsm8k_evaluation",
]
