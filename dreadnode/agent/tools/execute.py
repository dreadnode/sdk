import asyncio
import contextlib
import os
import sys

from loguru import logger

from dreadnode.agent.tools.base import tool


@tool(catch=True)
async def command(
    cmd: list[str],
    *,
    timeout: int = 120,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    input: str | None = None,
) -> str:
    """
    Execute a shell command.

    ## Best Practices
    - Argument Format: Command and arguments must be a list of strings.
    - No Shell Syntax: Does not use a shell (no pipes, redirection, var expansion, etc.).
    - Error on Failure: Raises RuntimeError for non-zero exit codes.
    - Use input Parameter: Send data to the command's standard input to avoid hanging.

    Args:
        cmd: The command to execute as a list of strings.
        timeout: Maximum execution time in seconds.
        cwd: The working directory for the command.
        env: Environment variables for the command.
        input: Optional string to send to the command's standard input.
    """
    command_str = " ".join(cmd)
    logger.debug(f"Executing '{command_str}'")

    process_env = os.environ.copy()
    if env:
        process_env.update(env)

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.PIPE if input is not None else None,
        env=process_env,
        cwd=cwd,
    )

    output = ""

    async def read_stdout() -> None:
        nonlocal output

        if not proc.stdout:
            return

        while True:
            chunk = await proc.stdout.read(1024)
            if not chunk:
                break
            output += chunk.decode(errors="replace")

    async def write_and_close_stdin() -> None:
        if proc.stdin and input:
            proc.stdin.write(input.encode())
            await proc.stdin.drain()
            proc.stdin.close()

    try:
        await asyncio.wait_for(
            asyncio.gather(read_stdout(), write_and_close_stdin()), timeout=timeout
        )
        await proc.wait()

    except asyncio.TimeoutError as e:
        error_message = f"Command '{command_str}' timed out after {timeout} seconds."
        if output:
            error_message += f"\n\nPartial Output:\n{output}"
        logger.warning(error_message)

        with contextlib.suppress(OSError):
            proc.kill()
            await proc.wait()

        raise TimeoutError(error_message) from e

    if proc.returncode != 0:
        logger.error(
            f"Command '{command_str}' failed with return code {proc.returncode}:\n{output}"
        )
        raise RuntimeError(f"Command failed ({proc.returncode}):\n{output}")

    logger.debug(f"Command '{command_str}' completed:\n{output}")
    return output


@tool(catch=True)
async def python(code: str, *, timeout: int = 120) -> str:
    """
    Execute Python code.

    This tool is ideal for tasks that require custom logic like loops and conditionals, \
    or for parsing and transforming the output from other tools. Use it to implement a \
    sequence of actions, perform file I/O, or create functionality not covered by other \
    available tools.

    ## Best Practices
    - Capture Output: Your script *must* print results to standard output (`print(...)`) to be captured.
    - Self-Contained: Import all required standard libraries (e.g., `os`, `json`) within the script.
    - Handle Errors: Write robust code. Unhandled exceptions in your script will cause the tool to fail.
    - String-Based I/O: Ensure all printed output can be represented as a string. Use formats like JSON (`json.dumps`) for complex data.

    Args:
        code: The Python code to execute as a string.
        timeout: Maximum time in seconds to allow for code execution.
    """
    try:
        logger.debug(f"Executing python:\n{code}")
        proc = await asyncio.create_subprocess_exec(
            *[sys.executable, "-"],
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=code.encode("utf-8")), timeout=timeout
        )
        output = stdout.decode(errors="ignore") + stderr.decode(errors="ignore")
    except asyncio.TimeoutError as e:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        raise TimeoutError(f"Execution timed out after {timeout} seconds") from e
    except Exception as e:
        logger.error(f"Error executing code in Python: {e}")
        raise

    if proc.returncode != 0:
        logger.error(f"Execution failed with return code {proc.returncode}:\n{output}")
        raise RuntimeError(f"Execution failed ({proc.returncode}):\n{output}")

    logger.debug(f"Execution successful. Output:\n{output}")
    return output
