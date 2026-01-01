"""Terminal-Bench Evaluation

This module provides a Terminal-Bench style evaluation where an agent
completes tasks in isolated Docker containers, scored by test scripts.

Following Terminal-Bench's design, the agent has a single bash tool
for interacting with the terminal - no separate file/command tools.
"""

import typing as t
from pathlib import Path

from loguru import logger

from dreadnode.core.task import task
from dreadnode.environments.docker import DockerEnvironment
from dreadnode.evaluations import Evaluation
from dreadnode.evaluations.terminal_bench.agent import create_terminal_agent
from dreadnode.evaluations.terminal_bench.dataset import (
    load_terminal_bench,
)
from dreadnode.evaluations.terminal_bench.scorers import (
    TerminalResult,
    efficiency_scorer,
    task_success_scorer,
    terminal_composite_scorer,
)


def _parse_dockerfile(dockerfile_content: str) -> tuple[str, list[str]]:
    """Parse a Dockerfile to extract base image and RUN commands.

    Returns:
        Tuple of (base_image, list_of_run_commands)
    """
    base_image = "python:3.12-slim"
    run_commands = []

    for line in dockerfile_content.split("\n"):
        line = line.strip()
        if line.startswith("FROM "):
            parsed_image = line[5:].strip()
            # Use fallback for Terminal-Bench private images
            if "ghcr.io/laude-institute/t-bench" in parsed_image:
                # Map to public equivalents
                if "python-3-13" in parsed_image:
                    base_image = "python:3.13-slim"
                elif "python-3-12" in parsed_image:
                    base_image = "python:3.12-slim"
                elif "python-3-11" in parsed_image:
                    base_image = "python:3.11-slim"
                elif "ubuntu" in parsed_image:
                    base_image = "ubuntu:24.04"
                else:
                    base_image = "python:3.12-slim"
            else:
                base_image = parsed_image
        elif line.startswith("RUN "):
            run_commands.append(line[4:].strip())
        elif line.startswith("WORKDIR "):
            # Convert WORKDIR to mkdir + cd commands
            workdir = line[8:].strip()
            run_commands.append(f"mkdir -p {workdir}")

    return base_image, run_commands


async def run_terminal_task(
    instruction: str,
    task_dir: str | None = None,
    difficulty: str = "medium",
    max_agent_timeout_sec: float = 900.0,
    max_test_timeout_sec: float = 180.0,
    model: str = "groq/moonshotai/kimi-k2-instruct-0905",
    max_steps: int = 30,
) -> TerminalResult:
    """
    Run a single Terminal-Bench task in a Docker container.

    Following Terminal-Bench's design:
    1. Parse Dockerfile to get base image and setup commands
    2. Copy task-deps into container
    3. Run agent with single bash tool
    4. Run test script to verify success
    5. Return TerminalResult with outcome

    Args:
        instruction: The task instruction for the agent.
        task_dir: Path to the task directory with Dockerfile, tests, etc.
        difficulty: Task difficulty level.
        max_agent_timeout_sec: Timeout for agent execution.
        max_test_timeout_sec: Timeout for test script execution.
        model: LLM model for the agent.
        max_steps: Maximum agent steps.

    Returns:
        TerminalResult with success status and metrics.
    """
    task_path = Path(task_dir) if task_dir else None

    # Parse Dockerfile to get base image and setup commands
    setup_commands: list[str] = []

    if task_path and (task_path / "Dockerfile").exists():
        dockerfile_content = (task_path / "Dockerfile").read_text()
        # Remove canary comment
        lines = dockerfile_content.split("\n")
        if lines and lines[0].startswith("# BENCHMARK DATA"):
            lines = lines[1:]
        dockerfile_content = "\n".join(lines)
        image_name, setup_commands = _parse_dockerfile(dockerfile_content)
    else:
        image_name = "python:3.12-slim"

    environment = DockerEnvironment(
        image=image_name,
        env={"TERM": "xterm"},
    )

    try:
        # Setup environment
        logger.info(f"Setting up Docker environment with image '{image_name}'")
        await environment.setup()

        # Run setup commands from Dockerfile (RUN instructions)
        if setup_commands:
            logger.debug(f"Running {len(setup_commands)} setup commands...")
            for cmd in setup_commands:
                exit_code, output = await environment.container.run(
                    cmd,
                    timeout=300,  # 5 min timeout for setup
                    stream_output=False,
                )
                if exit_code != 0:
                    logger.warning(f"Setup command failed: {cmd[:50]}... -> {output[:100]}")

        # Ensure /app directory exists
        await environment.container.run("mkdir -p /app", timeout=10, stream_output=False)

        # Copy task-deps into container if they exist
        if task_path:
            task_deps = task_path / "task-deps"
            if task_deps.exists() and task_deps.is_dir():
                logger.debug("Copying task-deps into container...")
                for dep_file in task_deps.rglob("*"):
                    if dep_file.is_file():
                        rel_path = dep_file.relative_to(task_deps)
                        container_path = f"/app/{rel_path}"
                        content = dep_file.read_bytes()
                        # Create parent dirs and write file
                        await environment.container.run(
                            f"mkdir -p /app/{rel_path.parent}",
                            timeout=10,
                            stream_output=False,
                        )
                        # Use base64 to safely transfer binary content
                        import base64

                        encoded = base64.b64encode(content).decode()
                        await environment.container.run(
                            f"echo '{encoded}' | base64 -d > {container_path}",
                            timeout=30,
                            stream_output=False,
                        )

        # Create agent with single bash tool (Terminal-Bench style)
        agent = create_terminal_agent(
            environment=environment,
            model=model,
            max_steps=max_steps,
        )

        logger.info(f"Running agent with instruction: {instruction[:100]}...")
        trajectory = await agent.run(instruction)
        trajectory_steps = len(trajectory.steps)
        logger.info(f"Agent completed in {trajectory_steps} steps")

        # Run test script
        logger.debug("Running test script...")

        # Determine test script to run
        if task_path and (task_path / "run-tests.sh").exists():
            # Copy run-tests.sh into container
            test_script_content = (task_path / "run-tests.sh").read_text()

            # Copy tests directory
            tests_dir = task_path / "tests"
            if tests_dir.exists():
                await environment.container.run("mkdir -p /tests", timeout=10, stream_output=False)
                for test_file in tests_dir.rglob("*"):
                    if test_file.is_file():
                        rel_path = test_file.relative_to(tests_dir)
                        try:
                            content = test_file.read_text()
                            # Escape content for shell
                            escaped = content.replace("'", "'\"'\"'")
                            await environment.container.run(
                                f"mkdir -p /tests/{rel_path.parent} && cat > /tests/{rel_path} << 'TESTEOF'\n{content}\nTESTEOF",
                                timeout=30,
                                stream_output=False,
                            )
                        except Exception:
                            pass  # Skip binary files

            # Write and run the test script
            escaped_script = test_script_content.replace("'", "'\"'\"'")
            await environment.container.run(
                f"cat > /run-tests.sh << 'SCRIPTEOF'\n{test_script_content}\nSCRIPTEOF",
                timeout=10,
                stream_output=False,
            )
            await environment.container.run(
                "chmod +x /run-tests.sh", timeout=10, stream_output=False
            )

            test_exit_code, test_output = await environment.container.run(
                "cd /app && TEST_DIR=/tests /run-tests.sh",
                timeout=int(max_test_timeout_sec),
                stream_output=False,
            )
        else:
            # No test script - just check if agent completed
            test_exit_code = 0
            test_output = "No test script provided"

        success = test_exit_code == 0
        if success:
            logger.success(f"Task PASSED: {test_output.strip()[-200:]}")
        else:
            logger.warning(f"Task FAILED (exit {test_exit_code}): {test_output.strip()[-200:]}")

        return TerminalResult(
            trajectory_steps=trajectory_steps,
            test_exit_code=test_exit_code,
            test_output=test_output,
            success=success,
        )

    finally:
        # Always teardown
        logger.debug("Tearing down Docker environment...")
        await environment.teardown()


def create_terminal_task(
    model: str = "groq/moonshotai/kimi-k2-instruct-0905",
    max_steps: int = 25,
) -> t.Callable[..., t.Awaitable[TerminalResult]]:
    """
    Create a task function for terminal evaluation.

    This wraps run_terminal_task with the model/max_steps config,
    making it compatible with the Evaluation framework.
    """

    @task(name="Terminal Task")
    async def terminal_task(
        instruction: str,
        task_dir: str | None = None,
        difficulty: str = "medium",
        max_agent_timeout_sec: float = 900.0,
        max_test_timeout_sec: float = 180.0,
    ) -> TerminalResult:
        return await run_terminal_task(
            instruction=instruction,
            task_dir=task_dir,
            difficulty=difficulty,
            max_agent_timeout_sec=max_agent_timeout_sec,
            max_test_timeout_sec=max_test_timeout_sec,
            model=model,
            max_steps=max_steps,
        )

    return terminal_task


def create_terminal_bench_evaluation(
    model: str = "groq/moonshotai/kimi-k2-instruct-0905",
    max_steps: int = 25,
    concurrency: int = 1,  # Sequential by default (Docker resource constraints)
    use_composite_scorer: bool = True,
    difficulty: str | list[str] | None = None,
    category: str | list[str] | None = None,
    limit: int | None = None,
) -> Evaluation[dict[str, t.Any], TerminalResult]:
    """
    Create a Terminal-Bench evaluation.

    Args:
        model: The LLM model to use for the agent.
        max_steps: Maximum steps per task.
        concurrency: Number of concurrent tasks (1 recommended for Docker).
        use_composite_scorer: Whether to use composite or individual scorers.
        difficulty: Filter tasks by difficulty ("easy", "medium", "hard").
        category: Filter tasks by category.
        limit: Maximum number of tasks to run.

    Returns:
        An Evaluation instance ready to run.

    Example:
        ```python
        # Run easy tasks only
        evaluation = create_terminal_bench_evaluation(difficulty="easy")
        result = await evaluation.run()

        # Run all 241 tasks
        evaluation = create_terminal_bench_evaluation()
        result = await evaluation.run()

        print(f"Pass rate: {result.pass_rate:.1%}")
        ```
    """
    if use_composite_scorer:
        scorers = [terminal_composite_scorer]
    else:
        scorers = [task_success_scorer, efficiency_scorer]

    # Load dataset with filters
    dataset = load_terminal_bench(
        difficulty=difficulty,
        category=category,
        limit=limit,
    )

    return Evaluation(
        name="Terminal-Bench",
        description=f"Terminal-Bench evaluation with {len(dataset)} tasks.",
        task=create_terminal_task(model=model, max_steps=max_steps),
        dataset=dataset,
        dataset_input_mapping={
            "instruction": "instruction",
            "task_dir": "task_dir",
            "difficulty": "difficulty",
            "max_agent_timeout_sec": "max_agent_timeout_sec",
            "max_test_timeout_sec": "max_test_timeout_sec",
        },
        scorers=scorers,
        assert_scores=["task_success"] if not use_composite_scorer else [],
        concurrency=concurrency,
        iterations=1,
        tags=["terminal", "docker", "agentic", "benchmark"],
    )


# Default evaluation instance for discovery (easy tasks only for quick testing)
TerminalBenchEvaluation = create_terminal_bench_evaluation(difficulty="easy", limit=10)


if __name__ == "__main__":
    import asyncio

    import dreadnode as dn
    from dreadnode.evaluations.terminal_bench.dataset import get_task_stats

    async def main() -> None:
        # Print dataset stats
        stats = get_task_stats()
        print("Terminal-Bench Dataset Stats:")
        print(f"  Total tasks: {stats['total']}")
        print(f"  By difficulty: {stats['by_difficulty']}")
        print(f"  Top categories: {list(stats['by_category'].items())[:5]}")
        print()

        dn.configure(
            server="http://localhost:8000",
            token="kkxgoNneuuPN0-lKWV3ItqEqIYbxUOsL",
            project="terminal-bench",
        )

        # Run evaluation on easy tasks
        evaluation = create_terminal_bench_evaluation(
            difficulty="easy",
            limit=5,
        )
        result = await evaluation.run()

        print(f"\n{'=' * 50}")
        print(f"Pass rate: {result.pass_rate:.1%}")
        print(f"Total samples: {len(result.samples)}")
        print(f"Passed: {result.passed_count}")
        print(f"Failed: {result.failed_count}")

        for sample in result.samples:
            status = "PASS" if sample.passed else "FAIL"
            task_id = sample.context.get("id", "unknown") if sample.context else "unknown"
            print(f"  [{status}] {task_id}")

    asyncio.run(main())
