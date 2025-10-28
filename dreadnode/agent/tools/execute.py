import asyncio
import contextlib
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
) -> str:
    """
    Execute a shell command.

    Use this tool to run system utilities and command-line programs (e.g., `ls`, `cat`, `grep`). \
    It is designed for straightforward, single-shot operations and returns the combined output and error streams.

    ## Best Practices
    - Argument Format: The command and its arguments *must* be provided as a \
    list of strings (e.g., `["ls", "-la", "/tmp"]`), not as a single string.
    - No Shell Syntax: Does not use a shell. Features like pipes (`|`), \
    redirection (`>`), and variable expansion (`$VAR`) are not supported.
    - Error on Failure: The tool will raise a `RuntimeError` if the command returns a non-zero exit code.

    Args:
        cmd: The command to execute, provided as a list of strings.
        timeout: Maximum time in seconds to allow for command execution.
        cwd: The working directory in which to execute the command.
        env: Optional environment variables to set for the command.
    """
    try:
        command_str = " ".join(cmd)
        logger.debug(f"Executing '{command_str}'")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=cwd,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        output = stdout.decode() + stderr.decode()
    except asyncio.TimeoutError as e:
        logger.warning(f"Command '{command_str}' timed out after {timeout} seconds.")
        with contextlib.suppress(OSError):
            proc.kill()
        raise TimeoutError(f"Command timed out after {timeout} seconds") from e
    except Exception as e:
        logger.error(f"Error executing '{command_str}': {e}")
        raise

    if proc.returncode != 0:
        logger.error(f"Command '{command_str}' failed with return code {proc.returncode}: {output}")
        raise RuntimeError(f"Command failed ({proc.returncode}): {output}")

    logger.debug(f"Command '{command_str}':\n{output}")
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
