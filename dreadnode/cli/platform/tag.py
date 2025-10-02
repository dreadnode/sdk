import platform

from packaging.version import Version

from dreadnode.cli.platform.constants import SUPPORTED_ARCHITECTURES


def add_tag_arch_suffix(tag: str) -> str:
    """
    Add architecture suffix to a tag if it doesn't already have one.

    Args:
        tag: The original tag string.
    """
    if any(tag.endswith(f"-{arch}") for arch in SUPPORTED_ARCHITECTURES):
        return tag  # Tag already has a supported architecture suffix

    arch = platform.machine()

    if arch in ["x86_64", "AMD64"]:
        arch = "amd64"
    elif arch in ["arm64", "aarch64", "ARM64"]:
        arch = "arm64"
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    return f"{tag}-{arch}"


def tag_to_semver(tag: str) -> Version:
    """
    Extract semantic version from a tag by removing architecture suffix.

    Args:
        tag: The tag string that may contain an architecture suffix.

    Returns:
        A packaging Version object representing the semantic version.
    """
    return Version(tag.split("-")[0].removeprefix("v"))
