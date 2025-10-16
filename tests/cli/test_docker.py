import pytest

from dreadnode.cli.docker import DockerImage


@pytest.mark.parametrize(
    ("input_str", "expected_repo", "expected_tag"),
    [
        ("ubuntu", "library/ubuntu", "latest"),
        ("postgres", "library/postgres", "latest"),
        ("alpine:3.18", "library/alpine", "3.18"),
        ("redis:7", "library/redis", "7"),
    ],
)
def test_docker_image_official_images(
    input_str: str, expected_repo: str, expected_tag: str
) -> None:
    """Tests parsing of official Docker Hub images (e.g., 'ubuntu')."""
    img = DockerImage(input_str)
    assert img.registry is None  # Should be None if not provided
    assert img.repository == expected_repo
    assert img.tag == expected_tag
    assert img.digest is None


@pytest.mark.parametrize(
    ("input_str", "expected_repo", "expected_tag"),
    [
        ("dreadnode/image", "dreadnode/image", "latest"),
        ("bitnami/postgresql", "bitnami/postgresql", "latest"),
        ("minio/minio:RELEASE.2023-03-20T20-16-18Z", "minio/minio", "RELEASE.2023-03-20T20-16-18Z"),
    ],
)
def test_docker_image_namespaced_images(
    input_str: str, expected_repo: str, expected_tag: str
) -> None:
    """Tests parsing of namespaced Docker Hub images (e.g., 'dreadnode/image')."""
    img = DockerImage(input_str)
    assert img.registry is None  # Should be None if not provided
    assert img.repository == expected_repo
    assert img.tag == expected_tag
    assert img.digest is None


@pytest.mark.parametrize(
    ("input_str", "expected_registry", "expected_repo", "expected_tag"),
    [
        ("gcr.io/google-samples/hello-app:1.0", "gcr.io", "google-samples/hello-app", "1.0"),
        ("ghcr.io/owner/image:tag", "ghcr.io", "owner/image", "tag"),
        ("localhost:5000/my-app", "localhost:5000", "my-app", "latest"),
        ("my.registry:1234/a/b/c:v1", "my.registry:1234", "a/b/c", "v1"),
    ],
)
def test_docker_image_with_custom_registry(
    input_str: str, expected_registry: str, expected_repo: str, expected_tag: str
) -> None:
    """Tests parsing of images with a full registry hostname."""
    img = DockerImage(input_str)
    assert img.registry == expected_registry
    assert img.repository == expected_repo
    assert img.tag == expected_tag
    assert img.digest is None


@pytest.mark.parametrize(
    ("input_str", "expected_repo", "expected_tag", "expected_digest"),
    [
        ("ubuntu@sha256:abc", "library/ubuntu", None, "sha256:abc"),
        ("dreadnode/image@sha256:123", "dreadnode/image", None, "sha256:123"),
        ("gcr.io/app/image@sha256:xyz", "app/image", None, "sha256:xyz"),
        ("ubuntu:22.04@sha256:456", "library/ubuntu", "22.04", "sha256:456"),
    ],
)
def test_docker_image_with_digest(
    input_str: str, expected_repo: str, expected_tag: str | None, expected_digest: str
) -> None:
    """Tests parsing of images with a digest."""
    img = DockerImage(input_str)
    assert img.repository == expected_repo
    assert img.tag == expected_tag
    assert img.digest == expected_digest


def test_docker_image_self_format_and_normalization() -> None:
    """Test that DockerImage can handle its own normalized string outputs."""
    img1 = DockerImage("ubuntu")
    assert str(img1) == "library/ubuntu:latest"

    img2 = DockerImage(str(img1))
    assert img2.registry is None
    assert img2.repository == "library/ubuntu"
    assert img2.tag == "latest"
    assert img1 == img2


def test_docker_image_whitespace_handling() -> None:
    """Test that leading/trailing whitespace is properly stripped."""
    img = DockerImage("  ubuntu:22.04  \n")
    assert img.repository == "library/ubuntu"
    assert img.tag == "22.04"
    assert str(img) == "library/ubuntu:22.04"


@pytest.mark.parametrize(
    "case",
    [
        "",  # Empty string
        "   ",  # Just whitespace
        "@sha256:123",  # Just a digest
    ],
)
def test_docker_image_invalid_formats(case: str) -> None:
    """Test that invalid formats raise ValueError."""
    with pytest.raises(ValueError, match="Invalid Docker image format"):
        DockerImage(case)


def test_docker_image_string_methods_inheritance() -> None:
    """Test that string methods from the str parent class work as expected."""
    img = DockerImage("ubuntu:22.04")
    assert str(img) == "library/ubuntu:22.04"
    assert img.upper() == "LIBRARY/UBUNTU:22.04"
    assert img.startswith("library")


def test_docker_image_comparisons() -> None:
    """Test comparison operations."""
    img1 = DockerImage("ubuntu")
    img2 = DockerImage("library/ubuntu:latest")
    img3 = DockerImage("postgres")
    img4 = DockerImage("docker.io/library/ubuntu:latest")

    assert img1 == img2
    assert img1 != img3
    assert img1 != img4  # These are now different, as one specifies a registry
    assert img1 == "library/ubuntu:latest"


def test_docker_image_with_method() -> None:
    """Tests the with_() method for creating modified copies."""
    original = DockerImage("gcr.io/project/image:1.0")

    # Change registry
    with_new_registry = original.with_(registry="ghcr.io")
    assert str(with_new_registry) == "ghcr.io/project/image:1.0"
    assert with_new_registry.registry == "ghcr.io"

    # Remove registry by setting to None
    with_no_registry = original.with_(registry=None)
    assert str(with_no_registry) == "project/image:1.0"
    assert with_no_registry.registry is None

    # Change tag and remove digest
    with_digest = DockerImage("gcr.io/project/image:1.0@sha256:abc")
    with_new_tag = with_digest.with_(tag="2.0-beta", digest=None)
    assert str(with_new_tag) == "gcr.io/project/image:2.0-beta"
    assert with_new_tag.tag == "2.0-beta"
    assert with_new_tag.digest is None

    # Ensure original object is not mutated
    assert str(original) == "gcr.io/project/image:1.0"
