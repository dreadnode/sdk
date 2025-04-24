import base64
import contextlib
import dataclasses
import datetime
import hashlib
import io
import json
import typing as t
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import Decimal
from enum import Enum
from functools import lru_cache
from ipaddress import (
    IPv4Address,
    IPv4Interface,
    IPv4Network,
    IPv6Address,
    IPv6Interface,
    IPv6Network,
)
from pathlib import PosixPath
from re import Pattern
from uuid import UUID

from dreadnode.types import JsonDict, JsonValue
from dreadnode.util import safe_repr

# Types

HandlerFunc = Callable[[t.Any, set[int]], tuple[JsonValue, JsonDict]]

# Constants

EMPTY_SCHEMA: JsonDict = {}
UNKNOWN_OBJECT_SCHEMA: JsonDict = {"type": "object", "x-python-datatype": "unknown"}


# Helpers

try:
    import attrs

    def _is_attrs_instance(_cls: type) -> bool:
        return attrs.has(_cls)

except ModuleNotFoundError:

    def _is_attrs_instance(_cls: type) -> bool:
        return False


# Specific handlers


def _handle_sequence(
    obj: Sequence[t.Any] | set[t.Any] | frozenset[t.Any],
    seen: set[int],
) -> tuple[JsonValue, JsonDict]:
    obj_type = type(obj)
    items_list = list(obj)

    with contextlib.suppress(TypeError):
        items_list.sort()  # sort if possible (e.g., for sets)

    serialized: list[JsonValue] = []
    item_schemas: list[JsonDict] = []

    non_empty_schemas_found = False

    for item in items_list:
        s_item, schema_item = _serialize(item, seen)
        serialized.append(s_item)
        item_schemas.append(schema_item)
        if schema_item != EMPTY_SCHEMA:
            non_empty_schemas_found = True

    schema: JsonDict = {"type": "array"}
    if obj_type != list:  # noqa: E721
        schema["title"] = obj_type.__name__
        type_name_map = {tuple: "tuple", set: "set", frozenset: "set", deque: "deque"}
        schema["x-python-datatype"] = type_name_map.get(obj_type, obj_type.__name__)

    if not items_list:  # if empty, basic array schema is sufficient
        return serialized, schema

    if not non_empty_schemas_found:
        first_item_type = type(items_list[0])
        if first_item_type in {str, int, float, bool, type(None)} and all(
            type(item) is first_item_type for item in items_list
        ):
            type_map = {
                str: "string",
                int: "integer",
                float: "number",
                bool: "boolean",
                type(None): "null",
            }
            schema["items"] = {"type": type_map[first_item_type]}

    else:
        # Check if all non-empty schemas are the same
        first_real_schema = next((s for s in item_schemas if s != EMPTY_SCHEMA), None)
        if first_real_schema and all(s in (first_real_schema, EMPTY_SCHEMA) for s in item_schemas):
            # All items conform to the same schema (or are primitives implicitly covered)
            schema["items"] = first_real_schema
        else:
            # Mixed schemas, use prefixItems (best for tuples, compromise for lists/sets)
            schema["prefixItems"] = item_schemas  # type: ignore [assignment]

    return serialized, schema


def _handle_mapping(
    obj: Mapping[t.Any, t.Any],
    seen: set[int],
) -> tuple[JsonValue, JsonDict]:
    serialized_dict: JsonDict = {}
    schema_properties: JsonDict = {}

    for key, value in obj.items():
        str_key = key if isinstance(key, str) else safe_repr(key)
        val_serialized, val_schema = _serialize(value, seen)
        serialized_dict[str_key] = val_serialized
        if val_schema != EMPTY_SCHEMA:
            schema_properties[str_key] = val_schema

    schema: JsonDict = {"type": "object"}
    if not isinstance(obj, dict):
        schema["title"] = obj.__class__.__name__
        schema["x-python-datatype"] = "Mapping"

    if schema_properties:
        schema["properties"] = schema_properties

    return serialized_dict, schema


def _handle_bytes(
    obj: bytes,
    _seen: set[int],
    schema_extras: JsonDict | None = None,
) -> tuple[JsonValue, JsonDict]:
    schema = {
        "type": "string",
        "x-python-datatype": "bytes",
    }

    if obj.__class__.__name__ != "bytes":
        schema["title"] = obj.__class__.__name__

    try:
        serialized = obj.decode()
        if not serialized.isprintable():
            raise ValueError("Non-printable characters found")  # noqa: TRY301
    except (UnicodeDecodeError, ValueError):
        serialized = base64.b64encode(obj).decode()
        schema["format"] = "base64"

    return serialized, {**schema, **(schema_extras or {})}


def _handle_bytearray(
    obj: bytearray,
    seen: set[int],
) -> tuple[JsonValue, JsonDict]:
    serialized, schema = _handle_bytes(bytes(obj), seen)
    schema["x-python-datatype"] = "bytearray"
    if obj.__class__.__name__ != "bytearray":
        schema["title"] = obj.__class__.__name__
    return serialized, schema


def _handle_enum(
    obj: Enum,
    seen: set[int],
) -> tuple[JsonValue, JsonDict]:
    enum_cls = obj.__class__
    serialized, _ = _serialize(obj.value, seen)  # Process the underlying value

    # Determine schema type based on enum values
    value_type = "object"
    if enum_values := [e.value for e in enum_cls]:
        first_val_type = type(enum_values[0])
        if all(isinstance(v, first_val_type) for v in enum_values):
            type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
            value_type = type_map.get(first_val_type, "object")

    # Get serialized representations of all possible enum values
    serialized_enum_values = []
    for e in enum_cls:
        s_enum_val, _ = _serialize(e.value, seen.copy())
        serialized_enum_values.append(s_enum_val)

    schema: JsonDict = {
        "type": value_type,
        "title": enum_cls.__name__,
        "x-python-datatype": "Enum",
        "enum": serialized_enum_values,
    }

    return serialized, schema


def _handle_datetime_iso(
    obj: datetime.date | datetime.datetime | datetime.time,
    _seen: set[int],
) -> tuple[JsonValue, JsonDict]:
    format_map = {
        datetime.datetime: "date-time",
        datetime.date: "date",
        datetime.time: "time",
    }
    return obj.isoformat(), {
        "type": "string",
        "format": format_map.get(type(obj), "unknown-datetime"),
    }


def _handle_timedelta(
    obj: datetime.timedelta,
    _seen: set[int],
) -> tuple[JsonValue, JsonDict]:
    return obj.total_seconds(), {
        "type": "number",
        "format": "time-delta-seconds",
        "x-python-datatype": "timedelta",
    }


def _handle_decimal(
    obj: Decimal,
    _seen: set[int],
) -> tuple[JsonValue, JsonDict]:
    return str(obj), {"type": "string", "format": "decimal"}


def _handle_str_based(
    obj: t.Any,
    _seen: set[int],
    schema_extras: JsonDict | None = None,
) -> tuple[JsonValue, JsonDict]:
    return str(obj), {"type": "string", **(schema_extras or {})}


def _handle_uuid(
    obj: UUID,
    seen: set[int],
) -> tuple[JsonValue, JsonDict]:
    return _handle_str_based(obj, seen, {"format": "uuid"})


def _handle_path(
    obj: PosixPath,
    seen: set[int],
) -> tuple[JsonValue, JsonDict]:
    return _handle_str_based(
        obj,
        seen,
        {"format": "path", "x-python-datatype": "PosixPath"},
    )


def _handle_pattern(
    obj: Pattern[t.Any],
    seen: set[int],
) -> tuple[JsonValue, JsonDict]:
    return _handle_str_based(obj.pattern, seen, {"format": "regex"})


def _handle_exception(obj: Exception, _seen: set[int]) -> tuple[JsonValue, JsonDict]:
    return _handle_str_based(
        obj,
        _seen,
        {"title": obj.__class__.__name__, "x-python-datatype": "Exception"},
    )


def _handle_range(obj: range, _seen: set[int]) -> tuple[JsonValue, JsonDict]:
    return list(obj), {"type": "array", "items": {"type": "integer"}, "x-python-datatype": "range"}


def _handle_custom_object(
    obj: t.Any,
    keys: Iterable[str],
    seen: set[int],
    datatype_name: str,
) -> tuple[JsonValue, JsonDict]:
    obj_type = type(obj)
    serialized_props: JsonDict = {}
    schema_properties: JsonDict = {}

    for key in keys:
        with contextlib.suppress(AttributeError):
            value = getattr(obj, key)
            s_value, schema_value = _serialize(value, seen)
            serialized_props[key] = s_value
            if schema_value != EMPTY_SCHEMA:
                schema_properties[key] = schema_value

    schema: JsonDict = {
        "type": "object",
        "title": obj_type.__name__,
        "x-python-datatype": datatype_name,
    }

    if schema_properties:
        schema["properties"] = schema_properties

    return serialized_props, schema


def _handle_dataclass(obj: t.Any, seen: set[int]) -> tuple[JsonValue, JsonDict]:
    keys = [f.name for f in dataclasses.fields(obj) if f.repr]
    return _handle_custom_object(obj, keys, seen, "dataclass")


def _handle_attrs(obj: t.Any, seen: set[int]) -> tuple[JsonValue, JsonDict]:
    import attrs

    keys = [f.name for f in attrs.fields(obj.__class__)]
    return _handle_custom_object(obj, keys, seen, "attrs")


def _handle_pydantic_model(obj: t.Any, _seen: set[int]) -> tuple[JsonValue, JsonDict]:
    import pydantic

    if not isinstance(obj, pydantic.BaseModel):
        return safe_repr(obj), UNKNOWN_OBJECT_SCHEMA

    schema: JsonDict = {
        "type": "object",
        "title": type(obj).__name__,
        "x-python-datatype": "pydantic.BaseModel",
    }

    with contextlib.suppress(Exception):
        schema = obj.model_json_schema()

    return obj.model_dump(mode="json"), schema


def _handle_numpy_array(
    obj: t.Any,
    seen: set[int],
) -> tuple[JsonValue, JsonDict]:
    import numpy  # noqa: ICN001

    if not isinstance(obj, numpy.ndarray):
        return safe_repr(obj), UNKNOWN_OBJECT_SCHEMA

    serialized, schema = _handle_bytes(obj.tobytes(), seen)

    schema["x-python-datatype"] = "numpy.ndarray"
    schema["x-numpy-dtype"] = str(obj.dtype)
    schema["x-numpy-shape"] = list(obj.shape)

    return serialized, schema


def _handle_pandas_dataframe(
    obj: t.Any,
    seen: set[int],
) -> tuple[JsonValue, JsonDict]:
    import pandas as pd

    if not isinstance(obj, pd.DataFrame):
        return safe_repr(obj), UNKNOWN_OBJECT_SCHEMA

    serialized, schema = _serialize(obj.to_dict(), seen)
    schema["x-python-datatype"] = "pandas.DataFrame"

    return serialized, schema


def _handle_pandas_series(
    obj: t.Any,
    seen: set[int],
) -> tuple[JsonValue, JsonDict]:
    import pandas as pd

    if not isinstance(obj, pd.Series):
        return safe_repr(obj), UNKNOWN_OBJECT_SCHEMA

    serialized, schema = _serialize(obj.tolist(), seen)
    schema["x-python-datatype"] = "pandas.Series"

    return serialized, schema


def _handle_pil_image(
    obj: t.Any,
    _seen: set[int],
) -> tuple[JsonValue, JsonDict]:
    import PIL.Image

    if not isinstance(obj, PIL.Image.Image):
        return safe_repr(obj), UNKNOWN_OBJECT_SCHEMA

    buffer = io.BytesIO()
    export_format = "PNG"

    if hasattr(obj, "format") and isinstance(obj.format, str):
        export_format = obj.format.lower()

    obj.save(buffer, format=export_format)

    return _handle_bytes(
        buffer.getvalue(),
        _seen,
        {
            "x-python-datatype": "PIL.Image",
            "format": export_format.lower(),
        },
    )


def _handle_pydub_audio_segment(
    obj: t.Any,
    _seen: set[int],
) -> tuple[JsonValue, JsonDict]:
    from pydub import AudioSegment  # type: ignore[import-untyped, unused-ignore, import-not-found]

    if not isinstance(obj, AudioSegment):
        return safe_repr(obj), UNKNOWN_OBJECT_SCHEMA

    # AudioSegment can be in different formats, but we will use WAV as a default
    # Since there is no way to get the format from the AudioSegment object, we will use WAV
    # as a default format for export. TODO: Add a way to get the format from the user via tags may be.
    export_format = "wav"
    # Raw audio data from AudioSegment class is in bytes format.
    raw_bytes_data = obj.raw_data
    schema = {
        "x-python-datatype": "pydub.AudioSegment",
        "format": export_format,
        "x-audio-sample-rate": obj.frame_rate,
        "x-audio-channels": obj.channels,
        "x-audio-sample-width": obj.sample_width,
    }

    return _handle_bytes(raw_bytes_data, _seen, schema)


def _handle_moviepy_video_clip(
    obj: t.Any,
    _seen: set[int],
) -> tuple[JsonValue, JsonDict]:
    import tempfile
    from pathlib import Path

    from moviepy import (  # type: ignore[import-untyped, unused-ignore, import-not-found]
        VideoFileClip,
    )

    if not isinstance(obj, VideoFileClip):
        return safe_repr(obj), UNKNOWN_OBJECT_SCHEMA

    # Infer format from filename if available
    export_format = "mp4"
    if getattr(obj, "filename", None):
        ext = Path(obj.filename).suffix.lstrip(".").lower()
        if ext:
            export_format = ext

    # Export video to temp file
    with tempfile.NamedTemporaryFile(suffix=f".{export_format}") as temp_file:
        obj.write_videofile(
            temp_file.name,
        )
        raw_bytes_data = Path(temp_file.name).read_bytes()

    schema = {
        "x-python-datatype": "moviepy.VideoFileClip",
        "format": export_format,
        "start": obj.start,
        "end": obj.end,
        "duration": obj.duration,
        "fps": obj.fps,
        "size": obj.size,
        "rotation": obj.rotation,
        "aspect_ratio": obj.aspect_ratio,
        "w": obj.w,
        "h": obj.h,
        "n_frames": obj.n_frames,
    }

    return _handle_bytes(raw_bytes_data, _seen, schema)


def _handle_dataset(obj: t.Any, _seen: set[int]) -> tuple[JsonValue, JsonDict]:
    import datasets  # type: ignore[import-untyped]

    if not isinstance(obj, datasets.Dataset):
        return safe_repr(obj), UNKNOWN_OBJECT_SCHEMA

    buffer = io.BytesIO()
    obj.to_parquet(buffer)

    return _handle_bytes(
        buffer.getvalue(),
        _seen,
        {
            "x-python-datatype": "datasets.Dataset",
            "format": "parquet",
        },
    )


@lru_cache(maxsize=1)
def _get_handlers() -> dict[type, HandlerFunc]:
    handlers: dict[type, HandlerFunc] = {
        list: _handle_sequence,
        tuple: _handle_sequence,
        set: _handle_sequence,
        frozenset: _handle_sequence,
        deque: _handle_sequence,
        dict: _handle_mapping,
        bytes: _handle_bytes,
        bytearray: _handle_bytearray,
        Enum: _handle_enum,
        Decimal: _handle_decimal,
        datetime.datetime: _handle_datetime_iso,
        datetime.date: _handle_datetime_iso,
        datetime.time: _handle_datetime_iso,
        datetime.timedelta: _handle_timedelta,
        UUID: _handle_uuid,
        PosixPath: _handle_path,
        Pattern: _handle_pattern,
        range: _handle_range,
        Exception: _handle_exception,
        IPv4Address: lambda o, s: _handle_str_based(o, s, {"format": "ipv4"}),
        IPv6Address: lambda o, s: _handle_str_based(o, s, {"format": "ipv6"}),
        IPv4Interface: lambda o, s: _handle_str_based(o, s, {"x-python-datatype": "IPv4Interface"}),
        IPv6Interface: lambda o, s: _handle_str_based(o, s, {"x-python-datatype": "IPv6Interface"}),
        IPv4Network: lambda o, s: _handle_str_based(o, s, {"x-python-datatype": "IPv4Network"}),
        IPv6Network: lambda o, s: _handle_str_based(o, s, {"x-python-datatype": "IPv6Network"}),
    }

    # Pydantic

    with contextlib.suppress(Exception):
        import pydantic

        handlers[pydantic.NameEmail] = lambda o, s: _handle_str_based(
            o,
            s,
            {"format": "email", "x-python-datatype": "pydantic.NameEmail"},
        )
        handlers[pydantic.SecretStr] = lambda _o, s: _handle_str_based(
            "***",
            s,
            {"x-python-datatype": "pydantic.SecretStr"},
        )
        handlers[pydantic.SecretBytes] = lambda _o, s: _handle_bytes(
            b"***",
            s,
            {"x-python-datatype": "pydantic.SecretBytes"},
        )
        handlers[pydantic.AnyUrl] = lambda o, s: _handle_str_based(
            o,
            s,
            {"format": "url", "x-python-datatype": "pydantic.AnyUrl"},
        )
        handlers[pydantic.BaseModel] = _handle_pydantic_model

    with contextlib.suppress(Exception):
        import numpy as np

        handlers[np.ndarray] = _handle_numpy_array
        handlers[np.floating] = lambda o, s: _serialize(float(o), s)
        handlers[np.integer] = lambda o, s: _serialize(int(o), s)
        handlers[np.bool_] = lambda o, s: _serialize(bool(o), s)
        handlers[np.str_] = lambda o, s: _handle_str_based(
            o,
            s,
            {"x-python-datatype": "numpy.str_"},
        )
        handlers[np.bytes_] = lambda o, s: _handle_bytes(
            o,
            s,
            {"x-python-datatype": "numpy.bytes_"},
        )

    with contextlib.suppress(Exception):
        import pandas as pd

        handlers[pd.DataFrame] = _handle_pandas_dataframe
        handlers[pd.Series] = _handle_pandas_series

    with contextlib.suppress(Exception):
        import PIL.Image

        handlers[PIL.Image.Image] = _handle_pil_image

    with contextlib.suppress(Exception):
        import datasets

        handlers[datasets.Dataset] = _handle_dataset

    with contextlib.suppress(Exception):
        from pydub import AudioSegment

        handlers[AudioSegment] = _handle_pydub_audio_segment

    with contextlib.suppress(Exception):
        from moviepy import VideoFileClip

        handlers[VideoFileClip] = _handle_moviepy_video_clip

    return handlers


# Core functions


def _serialize(obj: t.Any, seen: set[int] | None = None) -> tuple[JsonValue, JsonDict]:  # noqa: PLR0911
    # Primitives early

    if isinstance(obj, str | int | float | bool) or obj is None:
        return obj, EMPTY_SCHEMA

    # Cycle tracking

    seen = seen or set()

    obj_id = id(obj)
    if obj_id in seen:
        return "<circular reference>", {}

    seen = seen.copy()
    seen.add(obj_id)

    obj_type = type(obj)
    handlers = _get_handlers()

    with contextlib.suppress(Exception):
        # MRO-based lookup first

        for base in obj_type.__mro__:
            if base in handlers:
                handler = handlers[base]
                return handler(obj, seen)

        # Common collections

        if isinstance(obj, list | tuple | set | frozenset | deque):
            return _handle_sequence(obj, seen)

        if isinstance(obj, Mapping):
            return _handle_mapping(obj, seen)

        # Common struct types

        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return _handle_dataclass(obj, seen)

        if _is_attrs_instance(obj_type):
            return _handle_attrs(obj, seen)

        # Generic sequences (if not list/tuple/set/deque and no other handler matched)

        if isinstance(obj, Sequence):
            return _handle_sequence(obj, seen)

        # Common fallbacks

        if hasattr(obj, "to_dict"):
            return _serialize(obj.to_dict(), seen)

        if hasattr(obj, "asdict"):  # e.g., namedtuple
            return _serialize(obj.asdict(), seen)

    # Fallback to repr

    return safe_repr(obj), {
        "type": "string",
        "title": obj_type.__name__,
        "x-python-datatype": "unknown",
    }


@dataclasses.dataclass
class Serialized:
    data: JsonValue | None
    data_bytes: bytes | None
    data_len: int
    data_hash: str
    schema: JsonDict
    schema_hash: str


EMPTY_HASH = "0" * 16


def serialize(obj: t.Any) -> Serialized:
    """
    Serializes a Python object into a JSON-compatible structure and
    generates a corresponding JSON Schema, ensuring consistency between
    the serialization format and the schema.

    Args:
        obj: The Python object to process.

    Returns:
        An object containing the serialized data, schema, and their hashes.
    """
    serialized, schema = _serialize(obj)

    if isinstance(serialized, str | int | bool | float):
        serialized_bytes = str(serialized).encode()
    else:
        serialized_bytes = json.dumps(serialized, separators=(",", ":")).encode()

    schema_str = json.dumps(schema, separators=(",", ":"))

    data_hash = EMPTY_HASH
    if serialized is not None:
        data_hash = hashlib.sha1(serialized_bytes).hexdigest()[:16]  # noqa: S324 (using sha1 for speed)

    schema_hash = EMPTY_HASH
    if schema and schema != EMPTY_SCHEMA:
        schema_hash = hashlib.sha1(schema_str.encode()).hexdigest()[:16]  # noqa: S324

    return Serialized(
        data=serialized,
        data_bytes=serialized_bytes if serialized is not None else None,
        data_len=len(serialized_bytes) if serialized is not None else 0,
        data_hash=data_hash,
        schema=schema,
        schema_hash=schema_hash,
    )
