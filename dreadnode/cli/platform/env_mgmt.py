import subprocess  # nosec
import sys
import typing as t
from pathlib import Path

from dreadnode.cli.platform.constants import (
    PLATFORM_SERVICES,
)
from dreadnode.cli.platform.version import LocalVersion
from dreadnode.logging_ import print_error, print_info

LineTypes = t.Literal["variable", "comment", "empty"]


class _EnvLine(t.NamedTuple):
    """Represents a line in an .env file with its type and content."""

    line_type: LineTypes
    key: str | None = None
    value: str = ""
    original_line: str = ""


def _parse_env_lines(content: str) -> list[_EnvLine]:
    """
    Parse .env file content into structured lines preserving all formatting.

    Args:
        content (str): The content of the .env file

    Returns:
        List[EnvLine]: List of parsed lines with their types
    """
    lines = []

    for line in content.split("\n"):
        stripped = line.strip()

        if not stripped:
            # Empty line
            lines.append(_EnvLine("empty", original_line=line))
        elif stripped.startswith("#"):
            # Comment line
            lines.append(_EnvLine("comment", original_line=line))
        elif "=" in stripped:
            # Variable line
            key, value = stripped.split("=", 1)
            lines.append(_EnvLine("variable", key.strip(), value.strip(), line))
        else:
            # Treat as comment/invalid line to preserve it
            lines.append(_EnvLine("comment", original_line=line))

    return lines


def _extract_variables(lines: list[_EnvLine]) -> dict[str, str]:
    """
    Extract just the variables from parsed lines.

    Args:
        lines: List of parsed environment file lines.

    Returns:
        dict[str, str]: Dictionary mapping variable names to their values.
    """
    return {
        line.key: line.value
        for line in lines
        if line.line_type == "variable" and line.key is not None
    }


def _find_insertion_points(
    base_lines: list[_EnvLine], remote_lines: list[_EnvLine], new_vars: dict[str, str]
) -> dict[str, int]:
    """
    Find the best insertion points for new variables based on remote file structure.

    Args:
        base_lines: Lines from local file.
        remote_lines: Lines from remote file.
        new_vars: New variables to place.

    Returns:
        dict[str, int]: Dict mapping variable names to insertion indices in base_lines.
    """
    insertion_points = {}

    # Build a map of variable positions in the remote file
    remote_var_positions = {}
    remote_var_context = {}

    for i, line in enumerate(remote_lines):
        if line.line_type == "variable":
            remote_var_positions[line.key] = i
            # Capture context (preceding comment/section)
            context_lines: list[str] = []
            j = i - 1
            while j >= 0 and remote_lines[j].line_type in ["comment", "empty"]:
                if remote_lines[j].line_type == "comment":
                    context_lines.insert(0, remote_lines[j].original_line)
                    break  # Stop at first comment (section header)
                j -= 1
            remote_var_context[line.key] = context_lines

    # Build a map of variable positions in the local file
    local_var_positions = {}
    for i, line in enumerate(base_lines):
        if line.line_type == "variable":
            local_var_positions[line.key] = i

    # For each new variable, find the best insertion point
    for new_var in new_vars:
        if new_var not in remote_var_positions:
            # Variable not in remote, place at end
            insertion_points[new_var] = len(base_lines)
            continue

        remote_pos = remote_var_positions[new_var]

        # Find variables that appear before this one in the remote file
        preceding_vars = [
            var
            for var, pos in remote_var_positions.items()
            if pos < remote_pos and var in local_var_positions
        ]

        # Find variables that appear after this one in the remote file
        following_vars = [
            var
            for var, pos in remote_var_positions.items()
            if pos > remote_pos and var in local_var_positions
        ]

        if preceding_vars:
            # Place after the last preceding variable that exists locally
            last_preceding = max(preceding_vars, key=lambda v: local_var_positions[v])
            insertion_points[new_var] = local_var_positions[last_preceding] + 1
        elif following_vars:
            # Place before the first following variable that exists locally
            first_following = min(following_vars, key=lambda v: local_var_positions[v])
            insertion_points[new_var] = local_var_positions[first_following]
        else:
            # No context, place at end
            insertion_points[new_var] = len(base_lines)

    return insertion_points


def _reconstruct_env_content(  # noqa: PLR0912
    base_lines: list[_EnvLine], merged_vars: dict[str, str], updated_remote_lines: list[_EnvLine]
) -> str:
    """
    Reconstruct .env content preserving structure from base while applying merged variables.

    Args:
        base_lines: Parsed lines from the local file (for structure).
        merged_vars: Dictionary of merged variables.
        updated_remote_lines: Parsed lines from updated remote (for new additions).

    Returns:
        str: Reconstructed .env content.
    """
    result_lines: list[str] = []
    processed_keys = set()

    # Identify new variables that need to be inserted
    existing_keys = {line.key for line in base_lines if line.line_type == "variable"}
    new_vars = {k: v for k, v in merged_vars.items() if k not in existing_keys}

    # Find optimal insertion points for new variables
    insertion_points = _find_insertion_points(base_lines, updated_remote_lines, new_vars)

    # Group new variables by insertion point
    vars_by_insertion: dict[int, list[str]] = {}
    for var, insertion_idx in insertion_points.items():
        if insertion_idx not in vars_by_insertion:
            vars_by_insertion[insertion_idx] = []
        vars_by_insertion[insertion_idx].append(var)

    # Process base structure, inserting new variables at appropriate points
    for i, line in enumerate(base_lines):
        # Insert new variables that belong before this line
        if i in vars_by_insertion:
            # Add context comments if this is a new section
            added_section_break = False
            for var in vars_by_insertion[i]:
                # Check if we need a section break (empty line before new variables)
                if not added_section_break and result_lines and result_lines[-1].strip():
                    # Look for context from remote file
                    remote_context = None
                    for remote_line in updated_remote_lines:
                        if remote_line.line_type == "variable" and remote_line.key == var:
                            # Find preceding comment in remote file
                            remote_idx = updated_remote_lines.index(remote_line)
                            for j in range(remote_idx - 1, -1, -1):
                                if updated_remote_lines[j].line_type == "comment":
                                    remote_context = updated_remote_lines[j].original_line
                                    break
                                if updated_remote_lines[j].line_type == "variable":
                                    break
                            break

                    # Add section break with context comment if available
                    if remote_context:
                        result_lines.append("")  # Empty line
                        result_lines.append(remote_context)  # Section comment
                    elif i > 0 and base_lines[i - 1].line_type == "variable":
                        result_lines.append("")  # Just empty line for separation

                    added_section_break = True

                # Add the new variable
                result_lines.append(f"{var}={new_vars[var]}")
                processed_keys.add(var)

        # Process the current line
        if line.line_type == "variable":
            if line.key in merged_vars:
                # Keep the variable, potentially with updated value
                new_value = merged_vars[line.key]
                if line.value == new_value:
                    # Value unchanged, keep original formatting
                    result_lines.append(line.original_line)
                else:
                    # Value changed, reconstruct line maintaining original key formatting
                    original_key_part = line.original_line.split("=")[0]
                    result_lines.append(f"{original_key_part}={new_value}")
                processed_keys.add(line.key)
            # If key not in merged_vars, it was removed, so skip it
        else:
            # Preserve comments and empty lines
            result_lines.append(line.original_line)

    # Handle any remaining new variables (those that should go at the very end)
    end_insertion_idx = len(base_lines)
    if end_insertion_idx in vars_by_insertion:
        # if result_lines and result_lines[-1].strip():  # Add separator if needed
        #     result_lines.append("")
        result_lines.extend(
            [
                f"{var}={new_vars[var]}"
                for var in vars_by_insertion[end_insertion_idx]
                if var not in processed_keys
            ]
        )

    # Join lines
    return "\n".join(result_lines)


def create_default_env_files(current_version: LocalVersion) -> None:
    """Create default environment files for all services in the current version.

    Copies sample environment files to actual environment files if they don't exist,
    and creates a combined .env file from API and UI environment files.

    Args:
        current_version: The current local version schema containing service information.

    Raises:
        RuntimeError: If sample environment files are not found or .env file creation fails.
    """
    for service in PLATFORM_SERVICES:
        for image in current_version.images:
            if image.service == service:
                env_file_path = current_version.get_env_path_by_service(service)
                if not env_file_path.exists():
                    # copy the sample
                    sample_env_file_path = current_version.get_example_env_path_by_service(service)
                    if sample_env_file_path.exists():
                        env_file_path.write_text(sample_env_file_path.read_text())
                    else:
                        print_error(
                            f"Sample environment file for {service} not found at {sample_env_file_path}."
                        )
                        raise RuntimeError(
                            f"Sample environment file for {service} not found. Cannot configure {service}."
                        )


def open_env_file(filename: Path) -> None:
    """Open the specified environment file in the default editor.

    Args:
        filename: The path to the environment file to open.
    """
    if sys.platform == "darwin":
        cmd = ["open", "-t", filename.as_posix()]
    else:
        cmd = ["xdg-open", filename.as_posix()]
    try:
        subprocess.run(cmd, check=False)  # noqa: S603 # nosec
        print_info("Opened environment file.")
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to open environment file: {e}")


def write_overrides_env(path: Path, **kwargs: str) -> None:
    """
    Write a .env file at `path` using provided kwargs.

    Rules:
      - ENV var names are UPPER_SNAKE_CASE (key uppercased).
      - int/float -> unquoted.
      - bool -> true/false (unquoted).
      - None -> empty value.
      - other types -> stringified, quoted.

    Example:
      write_env(bar="baz", replace_this=1)
      # -> .env containing:
      # BAR="BAZ"
      # REPLACE_THIS=1
    """

    def encode_value(val: t.Any) -> str:
        if isinstance(val, bool):
            return "true" if val else "false"
        if val is None:
            return ""
        if isinstance(val, (int, float)):
            return str(val)
        # default: treat as string, uppercase and quote
        s = str(val)
        s = s.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace('"', '\\"')
        return f'"{s}"'

    def encode_key(key: t.Any) -> str:
        s = str(key).upper()
        return s.replace("-", "_")

    lines = []
    for key, val in kwargs.items():
        name = encode_key(key)
        lines.append(f"{name}={encode_value(val)}")

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def remove_overrides_env(path: Path) -> None:
    """Remove the overrides .env file if it exists.

    Args:
        path: The path to the overrides .env file.
    """
    if path.exists():
        path.unlink()


def build_env_file(path: Path, **kwargs: str | None) -> None:
    """Build a .env file at the specified path with given key-value pairs.

    Reads an existing .env file if present, merges with provided key-value pairs,
    and writes back the updated content preserving formatting.

    Does not delete any existing keys unless explicitly set to None.

    Args:
        path: The path to the .env file to create/update.
        **kwargs: Key-value pairs to include in the .env file. Use None to remove a key.
    """
    # Read current existing env file content if it exists
    current_local_content = path.read_text(encoding="utf-8") if path.exists() else ""

    # Parse and extract current variables
    local_lines = _parse_env_lines(current_local_content)
    # remove line types that are empty
    local_lines = [line for line in local_lines if line.line_type != "empty"]
    merged_vars = _extract_variables(local_lines)

    # Apply requested changes:
    # - value is None -> remove key if it exists
    # - otherwise set/overwrite to the provided value (as-is; caller controls quoting)
    for k, v in kwargs.items():
        key = k.upper().replace("-", "_")
        if v is None:
            merged_vars.pop(key, None)
        else:
            merged_vars[key] = v

    # remove any empty keys
    merged_vars = {k: v for k, v in merged_vars.items() if k}

    # Reconstruct file content:
    # We don't have a "remote" layout here; use the existing local layout as the guide
    # so comments/spacing/order are preserved, and new keys are appended sensibly.
    updated_content = _reconstruct_env_content(local_lines, merged_vars, local_lines)

    # Ensure trailing newline for POSIX-friendly files
    if updated_content and not updated_content.endswith("\n"):
        updated_content += "\n"

    # Write back to disk
    path.write_text(updated_content, encoding="utf-8")


def read_env_file(path: Path) -> dict[str, str]:
    """Read a .env file and return its contents as a dictionary.

    Args:
        path: The path to the .env file to read.

    Returns:
        A dictionary containing the environment variables defined in the .env file.
    """
    if not path.exists():
        return {}

    content = path.read_text(encoding="utf-8")
    return _extract_variables(_parse_env_lines(content))
