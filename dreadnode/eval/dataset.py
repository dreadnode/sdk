import csv
import json
import typing as t
from pathlib import Path

import yaml

from dreadnode.common_types import AnyDict

FileFormat = t.Literal["jsonl", "csv", "json", "yaml", "yml"]


def load_dataset(path: Path | str, *, file_format: FileFormat | None = None) -> list[AnyDict]:
    """
    Loads a list of objects from a file path, with support for JSONL, CSV, JSON, and YAML formats.

    Args:
        path: The path to the file to load.
        file_format: Optional format of the file. If not provided, it will be inferred from the file extension.

    Returns:
        A list of dictionaries representing the objects in the file.
    """
    path = Path(path)
    dataset: list[AnyDict] = []

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return dataset

    file_format = file_format or t.cast("FileFormat", path.suffix.lstrip(".").lower())
    if file_format not in t.get_args(FileFormat):
        raise ValueError(f"Unsupported file format: {file_format}")

    if file_format == "jsonl":
        dataset = [json.loads(line) for line in content.splitlines() if line.strip()]

    elif file_format == "csv":
        reader = csv.DictReader(content.splitlines())
        dataset = list(reader)

    elif file_format == "json":
        dataset = json.loads(content)
        if not isinstance(dataset, list):
            raise ValueError("JSON file must contain a list of objects.")

    elif file_format in {"yaml", "yml"}:
        dataset = yaml.safe_load(content)
        if not isinstance(dataset, list):
            raise ValueError("YAML file must contain a list of objects.")

    return dataset
