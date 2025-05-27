"""
Tests for the serialization module functionality.
"""

import datetime
import ipaddress
import json
import re
import uuid
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from pathlib import PosixPath
from typing import Any, Optional

import pytest

from dreadnode.serialization import (
    object_hash,
    serialize,
    serialize_and_hash,
    serialize_with_schema,
    type_to_schema,
)


class TestEnum(Enum):
    """Test enum for serialization tests."""

    VALUE1 = "value1"
    VALUE2 = "value2"


@dataclass
class TestDataClass:
    """Test dataclass for serialization tests."""

    name: str
    value: int
    optional: str | None = None


def test_serialize_primitive_types():
    """Test serialization of primitive types."""
    # Test primitive types
    assert serialize(5) == 5
    assert serialize("test") == "test"
    assert serialize(True) is True
    assert serialize(None) is None
    assert serialize(3.14) == 3.14


def test_serialize_decimal():
    """Test serialization of decimal values."""
    # Decimals should be converted to floats
    decimal_val = Decimal("3.14159")
    serialized = serialize(decimal_val)
    assert isinstance(serialized, float)
    assert serialized == 3.14159


def test_serialize_datetime():
    """Test serialization of datetime objects."""
    # Datetime should be converted to ISO format string
    now = datetime.datetime.now(datetime.timezone.utc)
    serialized = serialize(now)
    assert isinstance(serialized, str)
    assert serialized == now.isoformat()


def test_serialize_uuid():
    """Test serialization of UUID objects."""
    # UUID should be converted to string
    test_uuid = uuid.uuid4()
    serialized = serialize(test_uuid)
    assert isinstance(serialized, str)
    assert serialized == str(test_uuid)


def test_serialize_ip_addresses():
    """Test serialization of IP address objects."""
    # IP addresses should be converted to strings
    ipv4 = ipaddress.IPv4Address("192.168.1.1")
    serialized = serialize(ipv4)
    assert isinstance(serialized, str)
    assert serialized == "192.168.1.1"

    ipv6 = ipaddress.IPv6Address("::1")
    serialized = serialize(ipv6)
    assert isinstance(serialized, str)
    assert serialized == "::1"


def test_serialize_path():
    """Test serialization of Path objects."""
    # Path objects should be converted to strings
    path = PosixPath("/tmp/test")
    serialized = serialize(path)
    assert isinstance(serialized, str)
    assert serialized == "/tmp/test"


def test_serialize_regex():
    """Test serialization of regex patterns."""
    # Regex patterns should be converted to strings
    pattern = re.compile(r"test\d+")
    serialized = serialize(pattern)
    assert isinstance(serialized, str)
    assert serialized == r"test\d+"


def test_serialize_enum():
    """Test serialization of Enum values."""
    # Enum values should be serialized to their values
    enum_val = TestEnum.VALUE1
    serialized = serialize(enum_val)
    assert serialized == "value1"


def test_serialize_collections():
    """Test serialization of collection types."""
    # Test list
    assert serialize([1, 2, 3]) == [1, 2, 3]

    # Test tuple - should be converted to list
    assert serialize((1, 2, 3)) == [1, 2, 3]

    # Test set - should be converted to list
    assert serialize({1, 2, 3}) == [1, 2, 3]

    # Test dict
    assert serialize({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_serialize_nested_structures():
    """Test serialization of nested structures."""
    # Create a complex nested structure
    complex_obj = {
        "name": "test",
        "values": [1, 2.5, None, True],
        "metadata": {
            "created": datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
            "id": uuid.UUID("12345678-1234-5678-1234-567812345678"),
            "tags": {"tag1", "tag2"},
        },
    }

    # Expected serialized result
    expected = {
        "name": "test",
        "values": [1, 2.5, None, True],
        "metadata": {
            "created": "2023-01-01T00:00:00+00:00",
            "id": "12345678-1234-5678-1234-567812345678",
            "tags": ["tag1", "tag2"],  # Set converted to list
        },
    }

    # Test serialization
    serialized = serialize(complex_obj)
    assert serialized == expected

    # Verify it can be JSON serialized without errors
    json_str = json.dumps(serialized)
    assert isinstance(json_str, str)


def test_serialize_dataclass():
    """Test serialization of dataclasses."""
    # Create a dataclass instance
    test_obj = TestDataClass(name="test", value=42)

    # Serialize it
    serialized = serialize(test_obj)

    # Verify result
    assert isinstance(serialized, dict)
    assert serialized == {"name": "test", "value": 42, "optional": None}


def test_serialize_with_schema():
    """Test serialization with schema generation."""
    test_obj = {"name": "test", "value": 42, "tags": ["a", "b", "c"]}

    # Serialize with schema
    serialized, schema = serialize_with_schema(test_obj)

    # Verify serialized data
    assert serialized == test_obj

    # Verify schema
    assert isinstance(schema, dict)
    assert schema["type"] == "object"
    assert "properties" in schema
    assert schema["properties"]["name"]["type"] == "string"
    assert schema["properties"]["value"]["type"] == "integer"
    assert schema["properties"]["tags"]["type"] == "array"


def test_object_hash():
    """Test object hashing functionality."""
    # Test simple value hashing
    str_hash = object_hash("test string")
    assert isinstance(str_hash, str)
    assert len(str_hash) > 0

    # Hash should be deterministic - same input = same hash
    assert object_hash("test string") == str_hash

    # Different inputs should produce different hashes
    assert object_hash("different string") != str_hash

    # Complex objects should be hashable too
    complex_obj = {
        "name": "test",
        "values": [1, 2, 3],
        "date": datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
    }

    complex_hash = object_hash(complex_obj)
    assert isinstance(complex_hash, str)

    # Same complex object should produce same hash
    assert (
        object_hash(
            {
                "name": "test",
                "values": [1, 2, 3],
                "date": datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
            }
        )
        == complex_hash
    )


def test_serialize_and_hash():
    """Test combined serialization and hashing."""
    test_obj = TestDataClass(name="test", value=42)

    # Serialize and hash
    serialized, obj_hash, schema, schema_hash = serialize_and_hash(test_obj)

    # Verify serialized data
    assert serialized == {"name": "test", "value": 42, "optional": None}

    # Verify hashes
    assert isinstance(obj_hash, str)
    assert isinstance(schema_hash, str)

    # Verify schema
    assert isinstance(schema, dict)

    # Same input should produce same hashes
    s2, h2, schema2, sh2 = serialize_and_hash(test_obj)
    assert h2 == obj_hash
    assert sh2 == schema_hash


def test_circular_reference_handling():
    """Test handling of circular references in objects."""
    # Create an object with circular reference
    circular_obj: dict[str, Any] = {"name": "circular"}
    circular_obj["self_ref"] = circular_obj

    # This should not cause an infinite recursion
    with pytest.raises(ValueError, match="Circular reference detected"):
        serialize(circular_obj)


def test_type_to_schema():
    """Test schema generation from Python types."""
    # Test primitive types
    assert type_to_schema(int)["type"] == "integer"
    assert type_to_schema(str)["type"] == "string"
    assert type_to_schema(float)["type"] == "number"
    assert type_to_schema(bool)["type"] == "boolean"

    # Test container types
    list_schema = type_to_schema(list[int])
    assert list_schema["type"] == "array"
    assert list_schema["items"]["type"] == "integer"

    dict_schema = type_to_schema(dict[str, float])
    assert dict_schema["type"] == "object"
    assert dict_schema["additionalProperties"]["type"] == "number"

    # Test optional types
    optional_schema = type_to_schema(Optional[str])
    assert "string" in optional_schema["anyOf"][0]["type"]
    assert optional_schema["anyOf"][1]["type"] == "null"
