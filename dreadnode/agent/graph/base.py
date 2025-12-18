import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import yaml

Label = str
EdgeType = str
TripletKey = tuple[Label, Label]


@dataclass(frozen=True)
class Universe:
    labels: set[Label]
    allowed: dict[TripletKey, set[EdgeType]]
    label_aliases: dict[Label, Label] = field(
        default_factory=dict
    )  # optional: map external -> internal

    def merge(self, other: "Universe") -> "Universe":
        merged_labels = set(self.labels) | set(other.labels)
        merged_allowed: dict[TripletKey, set[EdgeType]] = {
            k: set(v) for k, v in self.allowed.items()
        }
        for k, v in other.allowed.items():
            merged_allowed.setdefault(k, set()).update(v)
        merged_aliases = {**self.label_aliases, **other.label_aliases}
        return Universe(labels=merged_labels, allowed=merged_allowed, label_aliases=merged_aliases)


def load_universe_yaml(path: str) -> Universe:
    with open(path) as f:
        spec = yaml.safe_load(f)
    labels = set(spec["labels"])
    allowed: dict[TripletKey, set[EdgeType]] = {}
    for src, edge, dst in spec["edges"]:
        allowed.setdefault((src, dst), set()).add(edge)
    aliases = dict(spec.get("aliases", {}))
    return Universe(labels=labels, allowed=allowed, label_aliases=aliases)


NODES_PD_DTYPES = {
    "uuid": "string[pyarrow]",
    "group_id": "string[pyarrow]",
    "label": "string[pyarrow]",
    "name": "string[pyarrow]",
    "created_at": "datetime64[ns, UTC]",
    "attributes": "string[pyarrow]",
}

EDGE_EVENTS_PD_DTYPES = {
    "group_id": "string[pyarrow]",
    "src_uuid": "string[pyarrow]",
    "dst_uuid": "string[pyarrow]",
    "type": "string[pyarrow]",
    "event_ts": "datetime64[ns, UTC]",
    "event_kind": "string[pyarrow]",
    "attributes": "string[pyarrow]",
}


def coerce_nodes_schema(
    universe: Universe, df: pd.DataFrame, *, allowed_labels: set[str] | None = None
) -> pd.DataFrame:
    allowed_labels = allowed_labels or universe.labels
    df = df.copy()
    for c in ("uuid", "label"):
        if c not in df:
            raise ValueError(f"nodes_df missing required column '{c}'")
    if "group_id" not in df:
        df["group_id"] = "default"
    if "name" not in df:
        df["name"] = ""
    if "created_at" not in df:
        df["created_at"] = pd.Timestamp(datetime.now(timezone.utc), tz="UTC")
    if "attributes" not in df:
        df["attributes"] = "{}"
    bad = df[~df["label"].isin(allowed_labels)]
    if len(bad):
        raise ValueError("Invalid labels:\n" + bad[["uuid", "label"]].to_string(index=False))
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["attributes"] = df["attributes"].apply(
        lambda x: json.dumps(x) if not isinstance(x, str) else x
    )
    return df.astype(
        {
            "uuid": "string[pyarrow]",
            "group_id": "string[pyarrow]",
            "label": "string[pyarrow]",
            "name": "string[pyarrow]",
            "attributes": "string[pyarrow]",
        }
    ).astype({"created_at": "datetime64[ns, UTC]"})


def coerce_edge_events_schema(
    universe: Universe,
    edges_df: pd.DataFrame,
    nodes_lookup: pd.DataFrame,
    *,
    allowed_map: dict[tuple[str, str], set[str]] | None = None,
) -> pd.DataFrame:
    allowed_map = allowed_map or universe.allowed
    df = edges_df.copy()
    for c in ("src_uuid", "dst_uuid", "type"):
        if c not in df:
            raise ValueError(f"edges_df missing required column '{c}'")
    if "group_id" not in df:
        df["group_id"] = "default"
    if "event_ts" not in df:
        df["event_ts"] = pd.Timestamp(datetime.now(timezone.utc), tz="UTC")
    if "event_kind" not in df:
        df["event_kind"] = "open"
    if "attributes" not in df:
        df["attributes"] = "{}"
    label_map = dict(
        zip(nodes_lookup["uuid"].astype(str), nodes_lookup["label"].astype(str), strict=False)
    )
    df["src_label"] = df["src_uuid"].map(label_map)
    df["dst_label"] = df["dst_uuid"].map(label_map)
    missing = df[df["src_label"].isna() | df["dst_label"].isna()]
    if len(missing):
        raise ValueError(
            "Edge references unknown uuids; add nodes first:\n"
            + missing[["src_uuid", "dst_uuid", "type"]].to_string(index=False)
        )

    def ok(row: pd.Series) -> bool:
        return row["type"] in allowed_map.get((row["src_label"], row["dst_label"]), set())

    bad = df[~df.apply(ok, axis=1)]
    if len(bad):
        raise ValueError(
            "Disallowed edges:\n"
            + bad[["src_uuid", "src_label", "type", "dst_label", "dst_uuid"]].to_string(index=False)
        )
    df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True)
    df["attributes"] = df["attributes"].apply(
        lambda x: json.dumps(x) if not isinstance(x, str) else x
    )
    df = df.astype(
        {
            "group_id": "string[pyarrow]",
            "src_uuid": "string[pyarrow]",
            "dst_uuid": "string[pyarrow]",
            "type": "string[pyarrow]",
            "event_kind": "string[pyarrow]",
            "attributes": "string[pyarrow]",
        }
    ).astype({"event_ts": "datetime64[ns, UTC]"})
    return df[["group_id", "src_uuid", "dst_uuid", "type", "event_ts", "event_kind", "attributes"]]
