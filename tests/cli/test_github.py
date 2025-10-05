import pytest

from dreadnode.cli.github import GithubRepo


def test_github_repo_simple_format() -> None:
    repo = GithubRepo("owner/repo")
    assert repo.namespace == "owner"
    assert repo.repo == "repo"
    assert repo.ref == "main"
    assert str(repo) == "owner/repo@main"


def test_github_repo_simple_format_with_ref() -> None:
    repo = GithubRepo("owner/repo/tree/develop")
    assert repo.namespace == "owner"
    assert repo.repo == "repo"
    assert repo.ref == "develop"
    assert str(repo) == "owner/repo@develop"


@pytest.mark.parametrize(
    "case",
    [
        "https://github.com/owner/repo",
        "http://github.com/owner/repo",
        "https://github.com/owner/repo.git",
    ],
)
def test_github_repo_https_url(case: str) -> None:
    repo = GithubRepo(case)
    assert repo.namespace == "owner"
    assert repo.repo == "repo"
    assert repo.ref == "main"
    assert str(repo) == "owner/repo@main"


@pytest.mark.parametrize(
    ("case", "expected_ref"),
    [
        ("https://github.com/owner/repo/tree/feature/custom-branch", "feature/custom-branch"),
        ("https://github.com/owner/repo/blob/feature/custom-branch", "feature/custom-branch"),
    ],
)
def test_github_repo_https_url_with_ref(case: str, expected_ref: str) -> None:
    repo = GithubRepo(case)
    assert repo.namespace == "owner"
    assert repo.repo == "repo"
    assert repo.ref == expected_ref
    assert str(repo) == f"owner/repo@{expected_ref}"


@pytest.mark.parametrize(
    "case",
    [
        "git@github.com:owner/repo",
        "git@github.com:owner/repo.git",
    ],
)
def test_github_repo_ssh_url(case: str) -> None:
    repo = GithubRepo(case)
    assert repo.namespace == "owner"
    assert repo.repo == "repo"
    assert repo.ref == "main"
    assert str(repo) == "owner/repo@main"


@pytest.mark.parametrize(
    ("case", "expected_ref"),
    [
        ("https://raw.githubusercontent.com/owner/repo/main", "main"),
        ("https://raw.githubusercontent.com/owner/repo/feature-branch", "feature-branch"),
        ("https://raw.githubusercontent.com/owner/repo/feature/branch", "feature/branch"),
    ],
)
def test_github_repo_raw_githubusercontent(case: str, expected_ref: str) -> None:
    repo = GithubRepo(case)
    assert repo.namespace == "owner"
    assert repo.repo == "repo"
    assert repo.ref == expected_ref
    assert str(repo) == f"owner/repo@{expected_ref}"


@pytest.mark.parametrize(
    ("input_str", "expected_ref"),
    [
        ("owner/repo/tree/feature/custom", "feature/custom"),
        ("owner/repo/releases/tag/v1.0.0", "v1.0.0"),
    ],
)
def test_github_repo_ref_handling(input_str: str, expected_ref: str) -> None:
    """Test handling of different reference formats"""
    repo = GithubRepo(input_str)
    assert repo.namespace == "owner"
    assert repo.repo == "repo"
    assert repo.ref == expected_ref
    assert repo.zip_url == f"https://github.com/owner/repo/zipball/{expected_ref}"


@pytest.mark.parametrize(
    "case",
    [
        "owner/repo.js",
        "https://github.com/owner/repo.js",
        "git@github.com:owner/repo.js.git",
    ],
)
def test_github_repo_with_dots(case: str) -> None:
    """Test repositories with dots in names"""
    repo = GithubRepo(case)
    assert repo.namespace == "owner"
    assert repo.repo == "repo.js"
    assert str(repo) == "owner/repo.js@main"


@pytest.mark.parametrize(
    "case",
    [
        "owner-name/repo-name",
        "https://github.com/owner-name/repo-name",
        "git@github.com:owner-name/repo-name.git",
    ],
)
def test_github_repo_with_dashes(case: str) -> None:
    """Test repositories with dashes in names"""
    repo = GithubRepo(case)
    assert repo.namespace == "owner-name"
    assert repo.repo == "repo-name"
    assert str(repo) == "owner-name/repo-name@main"


@pytest.mark.parametrize(
    "case",
    [
        "  owner/repo  ",
        "\nowner/repo\n",
        "\towner/repo\t",
    ],
)
def test_github_repo_whitespace_handling(case: str) -> None:
    """Test that whitespace is properly stripped"""
    repo = GithubRepo(case)
    assert repo.namespace == "owner"
    assert repo.repo == "repo"
    assert str(repo) == "owner/repo@main"


@pytest.mark.parametrize(
    "case",
    [
        "",  # Empty string
        "owner",  # Missing repo
        "owner/",  # Missing repo
        "/repo",  # Missing owner
        "owner/repo/extra",  # Too many parts
        "http://gitlab.com/owner/repo",  # Wrong domain
        "git@gitlab.com:owner/repo.git",  # Wrong domain
    ],
)
def test_github_repo_invalid_formats(case: str) -> None:
    """Test that invalid formats raise ValueError"""
    with pytest.raises(ValueError, match="Invalid GitHub repository format"):
        GithubRepo(case)


def test_github_repo_string_methods_inheritance() -> None:
    """Test that string methods work as expected"""
    repo = GithubRepo("owner/repo")
    assert repo.upper() == "OWNER/REPO@MAIN"
    assert repo.split("/") == ["owner", "repo@main"]
    assert repo.split("@") == ["owner/repo", "main"]
    assert repo.replace("owner", "newowner") == "newowner/repo@main"
    assert len(repo) == len("owner/repo@main")


def test_github_repo_comparisons() -> None:
    """Test comparison operations"""
    repo1 = GithubRepo("owner/repo")
    repo2 = GithubRepo("owner/repo")
    repo3 = GithubRepo("different/repo")

    assert repo1 == repo2
    assert repo1 != repo3
    assert repo1 == "owner/repo@main"


def test_github_repo_self_format() -> None:
    """Test that GithubRepo can handle its own string representations"""
    # Test basic format
    repo1 = GithubRepo("owner/repo@main")
    assert repo1.namespace == "owner"
    assert repo1.repo == "repo"
    assert repo1.ref == "main"
    assert str(repo1) == "owner/repo@main"

    # Test creating from existing repo string
    repo2 = GithubRepo(str(repo1))
    assert repo2.namespace == "owner"
    assert repo2.repo == "repo"
    assert repo2.ref == "main"
    assert str(repo2) == str(repo1)
