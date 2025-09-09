import pathlib

import ansible_runner

from dreadnode.constants import DEFAULT_TOOL_SEARCH_PATH


def find_playbook(
    tool_name: str,
    *,
    path: pathlib.Path = DEFAULT_TOOL_SEARCH_PATH,
) -> pathlib.Path | None:
    """Find the Ansible playbook for a given tool.

    Args:
        tool_name (str): The name of the tool.
        path (pathlib.Path): The directory to search in.

    Returns:
        pathlib.Path | None: The path to the playbook if found, else None.
    """
    playbook_path = path / "runner" / "playbooks" / f"{tool_name}-tool.yml"
    print(playbook_path)
    if playbook_path.exists():
        return playbook_path
    return None


def install_tool(tool_name: str, *, path: pathlib.Path = DEFAULT_TOOL_SEARCH_PATH) -> None:
    """Install a tool using Ansible.

    Args:
        package (str): The name of the package to install.
    """

    playbook = find_playbook(tool_name)

    if playbook is None:
        raise FileNotFoundError(f"No playbook found for tool '{tool_name}' in {path}")

    rc, status, stats_path = run_playbook(str(playbook), private_data_dir=path)


def run_playbook(
    playbook: str,
    *,
    private_data_dir: pathlib.Path = DEFAULT_TOOL_SEARCH_PATH / "runner",
    verbose: bool = False,
):
    """
    Run an Ansible playbook with ansible-runner and return (rc, status, stats_path).
    - private_data_dir: the runner dir (contains project/, inventory/, env/, etc.)
    - playbook: relative path under project/ (e.g., 'playbooks/tool-bbot.yml')
    - inventory/extravars/envvars: dicts or paths (runner accepts both)
    """
    r = ansible_runner.run(
        private_data_dir=str(private_data_dir),
        playbook=playbook,
        ident="install-tool",
        quiet=verbose,
        verbosity=1,
    )

    if r.status == "failed":
        raise RuntimeError(f"Playbook {playbook} failed (rc={r.rc})")

    return r.rc, r.status, r.config.artifact_dir
