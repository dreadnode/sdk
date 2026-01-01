"""Terminal-Bench evaluation for agentic terminal tasks.

This evaluation uses the full Terminal-Bench benchmark with 241 tasks
across various categories and difficulties.
"""

from dreadnode.evaluations.terminal_bench.dataset import (
    TerminalTask,
    get_task_stats,
    list_tasks,
    load_task,
    load_tasks,
    load_terminal_bench,
    load_terminal_bench_sample,
)
from dreadnode.evaluations.terminal_bench.evaluation import (
    TerminalBenchEvaluation,
    create_terminal_bench_evaluation,
    create_terminal_task,
    run_terminal_task,
)
from dreadnode.evaluations.terminal_bench.scorers import (
    TerminalResult,
    efficiency_scorer,
    task_success_scorer,
    terminal_composite_scorer,
)

__all__ = [
    # Evaluation
    "TerminalBenchEvaluation",
    "create_terminal_bench_evaluation",
    "create_terminal_task",
    "run_terminal_task",
    # Dataset
    "TerminalTask",
    "list_tasks",
    "load_task",
    "load_tasks",
    "load_terminal_bench",
    "load_terminal_bench_sample",
    "get_task_stats",
    # Scorers
    "TerminalResult",
    "task_success_scorer",
    "efficiency_scorer",
    "terminal_composite_scorer",
]
