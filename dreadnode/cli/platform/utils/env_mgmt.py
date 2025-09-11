import subprocess
import sys
import typing as t
from pathlib import Path

from dreadnode.cli.platform.constants import (
    SERVICES,
)
from dreadnode.cli.platform.schemas import LocalVersionSchema
from dreadnode.cli.platform.utils.printing import print_error, print_info

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
    """Extract just the variables from parsed lines.

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


def _merge_env_files(
    original_remote_content: str,
    current_local_content: str,
    updated_remote_content: str,
) -> dict[str, str]:
    """
    Merge .env files with the following logic:
    1. Local changes (updates/additions) take precedence over remote defaults
    2. Remote removals remove the key from local (unless locally modified)
    3. Remote additions are added to local
    4. Local additions are preserved

    Args:
        original_remote_content (str): Original remote .env content (baseline)
        current_local_content (str): Current local .env content (with local changes)
        updated_remote_content (str): Updated remote .env content (new remote state)

    Returns:
        Dict[str, str]: Merged variables dictionary
    """
    # Extract variables from each file
    original_remote = _extract_variables(_parse_env_lines(original_remote_content))
    current_local = _extract_variables(_parse_env_lines(current_local_content))
    updated_remote = _extract_variables(_parse_env_lines(updated_remote_content))

    # Result dictionary to build the merged content
    merged = {}

    # Step 1: Start with current local content (preserves local changes and additions)
    merged.update(current_local)

    # Step 2: Add new keys from updated remote (remote additions)
    merged.update(
        {
            key: value
            for key, value in updated_remote.items()
            if key not in original_remote
            and key not in current_local  # New remote addition not already locally added
        }
    )

    # Step 3: Handle remote removals
    for key in original_remote:
        # Only remove if the key was removed in remote and the local value matches the original remote value
        if (
            key not in updated_remote
            and key in current_local
            and current_local[key] == original_remote[key]
        ):
            merged.pop(key, None)

    # Step 4: Update values for keys that exist in both updated remote and weren't locally modified
    merged.update(
        {
            key: remote_value
            for key, remote_value in updated_remote.items()
            if (
                key in original_remote
                and key in current_local
                and current_local[key] == original_remote[key]
            )
        }
    )

    return merged


def _find_insertion_points(
    base_lines: list[_EnvLine], remote_lines: list[_EnvLine], new_vars: dict[str, str]
) -> dict[str, int]:
    """Find the best insertion points for new variables based on remote file structure.

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
    """Reconstruct .env content preserving structure from base while applying merged variables.

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
        if result_lines and result_lines[-1].strip():  # Add separator if needed
            result_lines.append("")
        result_lines.extend(
            [
                f"{var}={new_vars[var]}"
                for var in vars_by_insertion[end_insertion_idx]
                if var not in processed_keys
            ]
        )

    # Join lines
    return "\n".join(result_lines)


def merge_env_files_content(
    original_remote_content: str, current_local_content: str, updated_remote_content: str
) -> str:
    """Main function to merge .env file contents preserving formatting and structure.

    Args:
        original_remote_content: Original remote .env content.
        current_local_content: Current local .env content.
        updated_remote_content: Updated remote .env content.

    Returns:
        str: Merged .env file content with preserved formatting.
    """
    # Get the merged variables using the original logic
    merged_vars = _merge_env_files(
        original_remote_content, current_local_content, updated_remote_content
    )

    # Parse the local file structure to preserve its formatting
    local_lines = _parse_env_lines(current_local_content)
    updated_remote_lines = _parse_env_lines(updated_remote_content)

    # Reconstruct content preserving local structure but with merged variables
    return _reconstruct_env_content(local_lines, merged_vars, updated_remote_lines)


def create_default_env_files(current_version: LocalVersionSchema) -> None:
    """Create default environment files for all services in the current version.

    Copies sample environment files to actual environment files if they don't exist,
    and creates a combined .env file from API and UI environment files.

    Args:
        current_version: The current local version schema containing service information.

    Raises:
        RuntimeError: If sample environment files are not found or .env file creation fails.
    """
    for service in SERVICES:
        for image in current_version.images:
            if image.service == service:
                env_file_path = current_version.get_env_path_by_service(service)
                if not env_file_path.exists():
                    # copy the sample
                    sample_env_file_path = current_version.get_example_env_path_by_service(service)
                    if sample_env_file_path.exists():
                        print_info(f"Copying {sample_env_file_path} to {env_file_path}...")
                        env_file_path.write_text(sample_env_file_path.read_text())
                    else:
                        print_error(
                            f"Sample environment file for {service} not found at {sample_env_file_path}."
                        )
                        raise RuntimeError(
                            f"Sample environment file for {service} not found. Cannot configure {service}."
                        )
