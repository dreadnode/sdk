# utils.py
"""Utility functions for dataset management."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import xxhash


def compute_checksum(file_path: Path, algorithm: str = "xxhash64") -> str:
    """
    Compute checksum of a file efficiently.

    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use

    Returns:
        Hex digest of the checksum
    """
    if algorithm == "xxhash64":
        hasher = xxhash.xxh64()
    elif algorithm == "sha256":
        hasher = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    chunk_size = 8192 * 1024  # 8MB chunks
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def compute_data_checksum(data: bytes, algorithm: str = "xxhash64") -> str:
    """Compute checksum of bytes."""
    if algorithm == "xxhash64":
        return xxhash.xxh64(data).hexdigest()
    if algorithm == "sha256":
        return hashlib.sha256(data).hexdigest()
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def serialize_schema(schema: pa.Schema) -> str:
    """Serialize PyArrow schema to JSON string."""
    return json.dumps(
        {
            "fields": [
                {
                    "name": field.name,
                    "type": str(field.type),
                    "nullable": field.nullable,
                    "metadata": field.metadata,
                }
                for field in schema
            ],
            "metadata": schema.metadata,
        }
    )


def deserialize_schema(schema_json: str) -> pa.Schema:
    """Deserialize PyArrow schema from JSON string."""
    data = json.loads(schema_json)
    fields = []
    for field_data in data["fields"]:
        field_type = pa.type_for_alias(field_data["type"])
        if field_type is None:
            # Handle complex types
            field_type = eval(f"pa.{field_data['type']}")  # noqa: S307
        fields.append(
            pa.field(
                field_data["name"],
                field_type,
                nullable=field_data["nullable"],
                metadata=field_data.get("metadata"),
            )
        )
    return pa.schema(fields, metadata=data.get("metadata"))


def sanitize_name(name: str) -> str:
    """Sanitize a name for use in paths and registry names."""
    return name.lower().replace(" ", "-").replace("_", "-")


def parse_target(target: str) -> tuple[str, str, str, str]:
    """
    Parse OCI target into components.

    Args:
        target: Target string like 'registry.io/org/dataset:version'

    Returns:
        Tuple of (registry, org, dataset, version)
    """
    if "://" in target:
        target = target.split("://", 1)[1]

    if ":" in target:
        repo, version = target.rsplit(":", 1)
    else:
        repo = target
        version = "latest"

    parts = repo.split("/")
    if len(parts) < 3:
        raise ValueError(f"Invalid target format: {target}. Expected: registry/org/dataset:version")

    registry = parts[0]
    org = parts[1]
    dataset = "/".join(parts[2:])

    return registry, org, dataset, version
