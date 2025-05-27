"""
Tests for the artifact storage functionality.
"""

import os
import tempfile
from pathlib import Path

import fsspec
import pytest

from dreadnode.artifact.storage import ArtifactStorage


@pytest.fixture
def local_fs():
    """Create a local filesystem for testing."""
    return fsspec.filesystem("file")


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def artifact_storage(local_fs):
    """Create an ArtifactStorage instance with a local filesystem."""
    return ArtifactStorage(local_fs)


@pytest.fixture
def test_file(temp_dir):
    """Create a test file for storage."""
    test_file_path = temp_dir / "test_file.txt"
    with open(test_file_path, "w") as f:
        f.write("Test content for artifact storage")
    return test_file_path


def test_store_file(artifact_storage, test_file, temp_dir):
    """Test storing a file."""
    # Target location for the file
    target_key = str(temp_dir / "artifacts" / "stored_file.txt")

    # Store the file
    uri = artifact_storage.store_file(test_file, target_key)

    # Verify the file exists at the target location
    assert os.path.exists(target_key)

    # Verify the content was stored correctly
    with open(target_key) as f:
        content = f.read()
        assert content == "Test content for artifact storage"

    # Check that the returned URI is correct
    assert uri.endswith("stored_file.txt")


def test_store_file_idempotent(artifact_storage, test_file, temp_dir):
    """Test that storing the same file multiple times is idempotent."""
    # Target location
    target_key = str(temp_dir / "artifacts" / "idempotent_file.txt")

    # Store the file once
    uri1 = artifact_storage.store_file(test_file, target_key)

    # Get file modification time
    mtime1 = os.path.getmtime(target_key)

    # Modify the source file
    with open(test_file, "w") as f:
        f.write("Updated content that should not be stored")

    # Store again with the same target key
    uri2 = artifact_storage.store_file(test_file, target_key)

    # Get new modification time
    mtime2 = os.path.getmtime(target_key)

    # Verify that the file wasn't overwritten (modification time should be the same)
    assert mtime1 == mtime2

    # Verify the content was not updated
    with open(target_key) as f:
        content = f.read()
        assert content == "Test content for artifact storage"

    # Check that both URIs are the same
    assert uri1 == uri2


def test_store_file_different_targets(artifact_storage, test_file, temp_dir):
    """Test storing the same file to different target locations."""
    # First target location
    target_key1 = str(temp_dir / "artifacts" / "file1.txt")

    # Second target location
    target_key2 = str(temp_dir / "artifacts" / "file2.txt")

    # Store the file to both locations
    uri1 = artifact_storage.store_file(test_file, target_key1)
    uri2 = artifact_storage.store_file(test_file, target_key2)

    # Verify both files exist
    assert os.path.exists(target_key1)
    assert os.path.exists(target_key2)

    # Verify the content is the same in both locations
    with open(target_key1) as f1, open(target_key2) as f2:
        assert f1.read() == f2.read()

    # Check that the URIs are different
    assert uri1 != uri2
    assert uri1.endswith("file1.txt")
    assert uri2.endswith("file2.txt")


def test_store_binary_file(artifact_storage, temp_dir):
    """Test storing a binary file."""
    # Create a binary test file
    binary_file_path = temp_dir / "binary_file.bin"
    with open(binary_file_path, "wb") as f:
        f.write(b"\x00\x01\x02\x03\x04\xff\xfe\xfd")

    # Target location
    target_key = str(temp_dir / "artifacts" / "stored_binary.bin")

    # Store the binary file
    uri = artifact_storage.store_file(binary_file_path, target_key)

    # Verify the file exists
    assert os.path.exists(target_key)

    # Verify the binary content was stored correctly
    with open(target_key, "rb") as f:
        content = f.read()
        assert content == b"\x00\x01\x02\x03\x04\xff\xfe\xfd"

    # Check the URI
    assert uri.endswith("stored_binary.bin")


def test_store_large_file(artifact_storage, temp_dir):
    """Test storing a large file that would trigger multipart upload in real systems."""
    # Create a relatively large test file (1MB)
    large_file_path = temp_dir / "large_file.bin"
    with open(large_file_path, "wb") as f:
        f.write(b"0" * 1024 * 1024)  # 1MB of zeros

    # Target location
    target_key = str(temp_dir / "artifacts" / "stored_large.bin")

    # Store the large file
    uri = artifact_storage.store_file(large_file_path, target_key)

    # Verify the file exists
    assert os.path.exists(target_key)

    # Verify the file size
    assert os.path.getsize(target_key) == 1024 * 1024

    # Check the URI
    assert uri.endswith("stored_large.bin")
