"""
Tests for the Object module and related functionality.
"""

import tempfile
from pathlib import Path

from dreadnode.object import ObjectRef, ObjectUri, ObjectVal
from dreadnode.serialization import object_hash


def test_object_ref():
    """Test ObjectRef creation and properties."""
    ref = ObjectRef(
        name="test-object", label="test-label", hash="test-hash", attributes={"attr1": "value1"}
    )

    assert ref.name == "test-object"
    assert ref.label == "test-label"
    assert ref.hash == "test-hash"
    assert ref.attributes == {"attr1": "value1"}


def test_object_uri():
    """Test ObjectUri creation and properties."""
    uri = ObjectUri(
        hash="test-hash", schema_hash="schema-hash", uri="file:///path/to/file", size=1024
    )

    assert uri.hash == "test-hash"
    assert uri.schema_hash == "schema-hash"
    assert uri.uri == "file:///path/to/file"
    assert uri.size == 1024
    assert uri.type == "uri"


def test_object_val():
    """Test ObjectVal creation and properties."""
    val = ObjectVal(hash="test-hash", schema_hash="schema-hash", value={"key": "value"})

    assert val.hash == "test-hash"
    assert val.schema_hash == "schema-hash"
    assert val.value == {"key": "value"}
    assert val.type == "val"


def test_log_artifact(configured_instance):
    """Test logging an artifact."""
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".txt") as temp_file:
        temp_file.write(b"Test content")
        temp_file.flush()

        # Log the artifact
        obj_ref = configured_instance.log_artifact(
            name="test-artifact", path=temp_file.name, attributes={"attr1": "value1"}
        )

        assert isinstance(obj_ref, ObjectRef)
        assert obj_ref.name == "test-artifact"
        assert obj_ref.attributes == {"attr1": "value1"}

        # Verify the artifact was stored
        artifact_path = (
            Path(configured_instance.config.local_dir) / "artifacts" / f"{obj_ref.hash}.txt"
        )
        assert artifact_path.exists()
        with open(artifact_path) as f:
            assert f.read() == "Test content"


def test_link_objects(configured_instance):
    """Test linking objects."""
    # Create two objects
    obj1 = configured_instance.log_artifact(
        name="object1", value={"data": "value1"}, attributes={"attr1": "value1"}
    )

    obj2 = configured_instance.log_artifact(
        name="object2", value={"data": "value2"}, attributes={"attr2": "value2"}
    )

    # Link the objects
    configured_instance.link_objects(obj1, obj2)

    # This mostly tests that the function doesn't throw any exceptions
    # A more comprehensive test would check internal storage and API interactions


def test_object_hash():
    """Test object hashing functionality."""
    # Test various object types
    str_hash = object_hash("test string")
    dict_hash = object_hash({"key": "value"})
    list_hash = object_hash([1, 2, 3])

    # Ensure they generate different hashes
    assert str_hash != dict_hash
    assert str_hash != list_hash
    assert dict_hash != list_hash

    # Test determinism - same input should give same hash
    assert object_hash("test string") == str_hash
    assert object_hash({"key": "value"}) == dict_hash
    assert object_hash([1, 2, 3]) == list_hash
