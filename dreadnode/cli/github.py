import os
import pathlib
import re
import tempfile
import typing as t
import zipfile

import httpx
from rich.prompt import Prompt

from dreadnode.logging_ import confirm, console, print_info, print_warning
from dreadnode.user_config import UserConfig, find_dreadnode_saas_profiles, is_dreadnode_saas_server


class GithubRepo(str):  # noqa: SLOT000
    """
    A string subclass that normalizes various GitHub repository string formats.

    Supported formats:
    - Full URLs: https://github.com/owner/repo
    - SSH URLs: git@github.com:owner/repo.git
    - Simple format: owner/repo
    - With ref: owner/repo/tree/main
    - With complex ref: owner/repo/tree/feature/custom
    - With ref (URL): https://github.com/owner/repo/tree/main
    - With .git: owner/repo.git
    - Raw URLs: https://raw.githubusercontent.com/owner/repo/main
    - Release URLs: owner/repo/releases/tag/v1.0.0
    - ZIP URLs: https://github.com/owner/repo/zipball/main
    - Simple with ref: owner/repo@ref
    """

    # Instance properties
    namespace: str
    repo: str
    ref: str

    # Regex patterns
    SSH_PATTERN = re.compile(r"git@github\.com:([^/]+)/([^/]+?)(\.git)?$")
    SIMPLE_PATTERN = re.compile(r"^([^/]+)/([^/]+?)(\.git)?$")
    URL_PATTERN = re.compile(r"github\.com/([^/]+)/([^/]+?)(?:\.git|/(?:tree|blob)/(.+?))?$")
    RAW_PATTERN = re.compile(r"raw\.githubusercontent\.com/([^/]+)/([^/]+)/(.+)")
    RELEASE_PATTERN = re.compile(r"([^/]+)/([^/]+)/releases/tag/(.+)$")
    OWN_FORMAT_PATTERN = re.compile(r"^([^/]+)/([^/@:]+)@(.+)$")
    ZIPBALL_PATTERN = re.compile(r"github\.com/([^/]+)/([^/]+?)/zipball/(.+)$")

    def __new__(cls, value: t.Any, *_: t.Any, **__: t.Any) -> "GithubRepo":  # noqa: PLR0912, PLR0915
        if not isinstance(value, str):
            return super().__new__(cls, str(value))

        namespace = None
        repo = None
        ref = "main"

        value = value.strip()

        # Try our own format first (owner/repo@ref)
        match = cls.OWN_FORMAT_PATTERN.match(value)
        if match:
            namespace = match.group(1)
            repo = match.group(2)
            ref = match.group(3)

        # Try as an SSH URL
        elif value.startswith("git@"):
            match = cls.SSH_PATTERN.search(value)
            if match:
                namespace, repo = match.group(1), match.group(2)

        # Try as a full URL
        elif value.startswith(("http://", "https://")):
            url_parts = value.split("//", 1)[1]

            # Try zipball pattern first
            match = cls.ZIPBALL_PATTERN.search(url_parts)
            if match:
                namespace = match.group(1)
                repo = match.group(2)
                ref = match.group(3)

            # Try raw githubusercontent pattern
            elif url_parts.startswith("raw.githubusercontent.com"):
                match = cls.RAW_PATTERN.search(url_parts)
                if match:
                    namespace, repo, ref = match.group(1), match.group(2), match.group(3)

            # Try standard GitHub URL pattern
            else:
                match = cls.URL_PATTERN.search(url_parts)
                if match:
                    namespace = match.group(1)
                    repo = match.group(2)
                    ref = match.group(3) or ref

        # Try release tag format
        elif "/releases/tag/" in value:
            match = cls.RELEASE_PATTERN.match(value)
            if match:
                namespace, repo, ref = match.group(1), match.group(2), match.group(3)

        # Try simple owner/repo format
        else:
            # First try to extract any ref
            tree_parts = value.split("/tree/")
            blob_parts = value.split("/blob/")

            if len(tree_parts) > 1:
                value, ref = tree_parts[0], tree_parts[1]
            elif len(blob_parts) > 1:
                value, ref = blob_parts[0], blob_parts[1]

            # Now check for owner/repo pattern
            match = cls.SIMPLE_PATTERN.match(value)
            if match:
                namespace, repo = match.group(1), match.group(2)

        if not namespace or not repo:
            raise ValueError(f"Invalid GitHub repository format: {value}")

        repo = repo.removesuffix(".git")

        obj = super().__new__(cls, f"{namespace}/{repo}@{ref}")

        obj.namespace = namespace
        obj.repo = repo
        obj.ref = ref

        return obj

    @property
    def zip_url(self) -> str:
        """ZIP archive URL for the repository."""
        return f"https://github.com/{self.namespace}/{self.repo}/zipball/{self.ref}"

    @property
    def api_zip_url(self) -> str:
        """API ZIP archive URL for the repository."""
        return f"https://api.github.com/repos/{self.namespace}/{self.repo}/zipball/{self.ref}"

    @property
    def tree_url(self) -> str:
        """URL to view the tree at this reference."""
        return f"https://github.com/{self.namespace}/{self.repo}/tree/{self.ref}"

    @property
    def exists(self) -> bool:
        """Check if a repo exists (or is private) on GitHub."""
        response = httpx.get(f"https://github.com/{self.namespace}/{self.repo}")
        return response.status_code == 200

    def __repr__(self) -> str:
        return f"GithubRepo(namespace='{self.namespace}', repo='{self.repo}', ref='{self.ref}')"


def get_repo_archive_source_path(source_dir: pathlib.Path) -> pathlib.Path:
    """Return the actual source directory from a git repositoryZIP archive."""

    if not (source_dir / "Dockerfile").exists() and not (source_dir / "Dockerfile.j2").exists():
        # if src has been downloaded from a ZIP archive, it may contain a single
        # '<user>-<repo>-<commit hash>' folder, that is the actual source we want to use.
        # Check if source_dir contains only one folder and update it if so.
        children = list(source_dir.iterdir())
        if len(children) == 1 and children[0].is_dir():
            source_dir = children[0]

    return source_dir


def download_and_unzip_archive(url: str, *, headers: dict[str, str] | None = None) -> pathlib.Path:
    """
    Downloads a ZIP archive from the given URL and unzips it into a temporary directory.
    """

    temp_dir = pathlib.Path(tempfile.mkdtemp())
    local_zip_path = temp_dir / "archive.zip"

    print_info(f"Downloading {url} ...")

    # download to temporary file
    with httpx.stream("GET", url, follow_redirects=True, verify=True, headers=headers) as response:
        response.raise_for_status()
        with local_zip_path.open("wb") as zip_file:
            for chunk in response.iter_bytes(chunk_size=8192):
                zip_file.write(chunk)

    # unzip to temporary directory
    try:
        with zipfile.ZipFile(local_zip_path, "r") as zf:
            for member in zf.infolist():
                file_path = os.path.realpath(temp_dir / member.filename)
                if file_path.startswith(os.path.realpath(temp_dir)):
                    zf.extract(member, temp_dir)
                else:
                    raise RuntimeError("Invalid file path detected in archive")

    finally:
        # always remove the zip file
        if local_zip_path.exists():
            local_zip_path.unlink()

    return temp_dir


def validate_server_for_clone(user_config: UserConfig, current_profile: str | None) -> str | None:
    """
    Validate the server configuration for git clone operations.

    Returns:
        The profile name to use, or None if the user cancelled.
    """
    config = user_config.get_server_config(current_profile)
    current_server = config.url

    # If current server is a Dreadnode SaaS server, all good
    if is_dreadnode_saas_server(current_server):
        return current_profile or user_config.active_profile_name

    # Current server is not a Dreadnode SaaS server - warn user
    print_warning(
        f"Current server is not a Dreadnode SaaS server\n"
        f"  Current server:  [cyan]{current_server}[/]\n"
        f"  Current profile: [cyan]{current_profile or user_config.active_profile_name}[/]\n\n"
        "Git clone for private dreadnode repositories requires a Dreadnode SaaS server\n"
        "(ending with '.dreadnode.io') for authentication to work properly.\n"
    )

    # Check if there are any SaaS profiles available
    saas_profiles = find_dreadnode_saas_profiles(user_config)

    if saas_profiles:
        print_info("Available Dreadnode SaaS profiles:")
        for profile in saas_profiles:
            server_url = user_config.servers[profile].url
            console.print(f"  - [bold]{profile}[/] ({server_url})")

        choices = ["continue", "switch", "cancel"]
        choice = Prompt.ask(
            "\nChoose an option", choices=choices, default="cancel", show_choices=True
        )

        if choice == "continue":
            print_warning("Continuing with current server - private repository access may fail")
            return current_profile or user_config.active_profile_name
        if choice == "cancel":
            return None
        if choice == "switch":
            # Let user pick a profile
            profile_choice = Prompt.ask(
                "\nSelect profile to use",
                choices=saas_profiles,
                default=saas_profiles[0],
                console=console,
            )
            print_info(f"Using profile '[cyan]{profile_choice}[/]' for this operation")
            return profile_choice
    else:
        # No SaaS profiles available
        if not confirm("Continue anyway?"):
            print_info(
                "Cancelled. Use [bold]dreadnode login --server https://platform.dreadnode.io[/] to add a SaaS profile."
            )
        print_warning("Continuing with current server - private repository access may fail")
        return current_profile or user_config.active_profile_name

    return None
