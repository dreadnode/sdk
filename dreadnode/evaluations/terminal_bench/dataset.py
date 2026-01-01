"""Terminal-Bench dataset loader.

Loads tasks from the tasks/ directory which contains the full Terminal-Bench
benchmark with 241 tasks across various categories and difficulties.
"""

import typing as t
from dataclasses import dataclass, field
from pathlib import Path

import yaml

TASKS_DIR = Path(__file__).parent / "tasks"


@dataclass
class TerminalTask:
    """A single terminal task from Terminal-Bench."""

    id: str
    instruction: str
    difficulty: str = "medium"
    category: str = "general"
    tags: list[str] = field(default_factory=list)
    dockerfile: str = ""
    run_tests_script: str = ""
    test_files: dict[str, str] = field(default_factory=dict)
    task_deps_files: list[str] = field(default_factory=list)
    max_agent_timeout_sec: float = 900.0
    max_test_timeout_sec: float = 180.0
    parser_name: str = "pytest"
    task_dir: Path | None = None


def _remove_canary(content: str) -> str:
    """Remove Terminal-Bench canary comment from content."""
    lines = content.split("\n")
    if lines and lines[0].startswith("# BENCHMARK DATA"):
        lines = lines[1:]
    return "\n".join(lines)


def load_task(task_id: str) -> TerminalTask:
    """Load a single task by ID.

    Args:
        task_id: The task directory name (e.g., "hello-world").

    Returns:
        TerminalTask with all task data.

    Raises:
        FileNotFoundError: If the task doesn't exist.
    """
    task_dir = TASKS_DIR / task_id
    if not task_dir.exists():
        raise FileNotFoundError(f"Task not found: {task_id}")

    task_yaml = task_dir / "task.yaml"
    if not task_yaml.exists():
        raise FileNotFoundError(f"task.yaml not found for: {task_id}")

    # Parse task.yaml
    content = _remove_canary(task_yaml.read_text())
    task_data = yaml.safe_load(content)

    # Load Dockerfile
    dockerfile = task_dir / "Dockerfile"
    dockerfile_content = ""
    if dockerfile.exists():
        dockerfile_content = _remove_canary(dockerfile.read_text()).strip()

    # Load run-tests.sh
    run_tests = task_dir / "run-tests.sh"
    run_tests_content = ""
    if run_tests.exists():
        run_tests_content = run_tests.read_text()

    # List task-deps files
    task_deps = task_dir / "task-deps"
    task_deps_files = []
    if task_deps.exists() and task_deps.is_dir():
        for dep_file in task_deps.rglob("*"):
            if dep_file.is_file():
                task_deps_files.append(str(dep_file.relative_to(task_deps)))

    # Load test files
    tests_dir = task_dir / "tests"
    test_files = {}
    if tests_dir.exists():
        for test_file in tests_dir.rglob("*"):
            if test_file.is_file():
                rel_path = str(test_file.relative_to(tests_dir))
                try:
                    test_files[rel_path] = test_file.read_text()
                except Exception:
                    test_files[rel_path] = "[binary file]"

    return TerminalTask(
        id=task_id,
        instruction=task_data.get("instruction", "").strip(),
        difficulty=task_data.get("difficulty", "medium"),
        category=task_data.get("category", "general"),
        tags=task_data.get("tags", []),
        dockerfile=dockerfile_content,
        run_tests_script=run_tests_content,
        test_files=test_files,
        task_deps_files=task_deps_files,
        max_agent_timeout_sec=task_data.get("max_agent_timeout_sec", 900.0),
        max_test_timeout_sec=task_data.get("max_test_timeout_sec", 180.0),
        parser_name=task_data.get("parser_name", "pytest"),
        task_dir=task_dir,
    )


def list_tasks() -> list[str]:
    """List all available task IDs.

    Returns:
        List of task directory names.
    """
    if not TASKS_DIR.exists():
        return []

    return sorted(d.name for d in TASKS_DIR.iterdir() if d.is_dir() and (d / "task.yaml").exists())


def load_tasks(
    difficulty: str | list[str] | None = None,
    category: str | list[str] | None = None,
    tags: list[str] | None = None,
    limit: int | None = None,
) -> list[TerminalTask]:
    """Load tasks with optional filtering.

    Args:
        difficulty: Filter by difficulty ("easy", "medium", "hard") or list.
        category: Filter by category or list of categories.
        tags: Filter by tags (tasks must have all specified tags).
        limit: Maximum number of tasks to return.

    Returns:
        List of TerminalTask objects matching the filters.
    """
    # Normalize filters
    if isinstance(difficulty, str):
        difficulty = [difficulty]
    if isinstance(category, str):
        category = [category]

    tasks = []
    for task_id in list_tasks():
        task = load_task(task_id)

        # Apply filters
        if difficulty and task.difficulty not in difficulty:
            continue
        if category and task.category not in category:
            continue
        if tags and not all(t in task.tags for t in tags):
            continue

        tasks.append(task)

        if limit and len(tasks) >= limit:
            break

    return tasks


def load_terminal_bench(
    difficulty: str | list[str] | None = None,
    category: str | list[str] | None = None,
    tags: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, t.Any]]:
    """Load Terminal-Bench tasks as dictionaries for evaluation.

    Args:
        difficulty: Filter by difficulty ("easy", "medium", "hard") or list.
        category: Filter by category or list of categories.
        tags: Filter by tags (tasks must have all specified tags).
        limit: Maximum number of tasks to return.

    Returns:
        List of task dictionaries compatible with Evaluation.
        Only essential fields are included to keep context clean.
    """
    tasks = load_tasks(
        difficulty=difficulty,
        category=category,
        tags=tags,
        limit=limit,
    )

    return [
        {
            # Task identification (goes to context)
            "id": task.id,
            "difficulty": task.difficulty,
            "category": task.category,
            # Task inputs (mapped to task parameters)
            "instruction": task.instruction,
            "task_dir": str(task.task_dir) if task.task_dir else None,
            "max_agent_timeout_sec": task.max_agent_timeout_sec,
            "max_test_timeout_sec": task.max_test_timeout_sec,
        }
        for task in tasks
    ]


# Backwards compatibility
def load_terminal_bench_sample() -> list[dict[str, t.Any]]:
    """Load a sample of easy Terminal-Bench tasks for quick testing."""
    return load_terminal_bench(difficulty="easy", limit=10)


def get_task_stats() -> dict[str, t.Any]:
    """Get statistics about available tasks.

    Returns:
        Dictionary with task counts by difficulty and category.
    """
    tasks = load_tasks()

    difficulties: dict[str, int] = {}
    categories: dict[str, int] = {}

    for task in tasks:
        difficulties[task.difficulty] = difficulties.get(task.difficulty, 0) + 1
        categories[task.category] = categories.get(task.category, 0) + 1

    return {
        "total": len(tasks),
        "by_difficulty": difficulties,
        "by_category": dict(sorted(categories.items(), key=lambda x: -x[1])),
    }
