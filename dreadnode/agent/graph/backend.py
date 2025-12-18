from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

pd.options.mode.copy_on_write = True


@dataclass
class GroupPaths:
    base: Path
    nodes: Path
    edge_events: Path


class PandasTemporalStore:
    """
    Parquet-backed temporal graph store:
    - nodes.parquet: uuid, group_id, label, name, created_at, attributes
    - edge_events.parquet: group_id, src_uuid, dst_uuid, type, event_ts, event_kind, attributes

    Derives intervals [valid_from, valid_to) via groupby+shift(-1).
    """

    def __init__(self, root: str = "./lake_pandas"):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, group_id: str) -> GroupPaths:
        """
        Returns paths for the given group_id.
        If the group_id is "default", it will use the root directory.
        """
        base = self.root / (group_id or "default")
        base.mkdir(parents=True, exist_ok=True)
        return GroupPaths(
            base=base, nodes=base / "nodes.parquet", edge_events=base / "edge_events.parquet"
        )

    def _append_parquet(self, path: Path, df: pd.DataFrame) -> None:
        """
        Append a DataFrame to a Parquet file.
        """
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path, existing_data_behavior="append")

    def add_entities_df(
        self,
        *,
        group_id: str,
        nodes_df: pd.DataFrame | None = None,
        edges_df: pd.DataFrame | None = None,
        default_ts: datetime | None = None,
    ) -> None:
        """
        Add nodes and edges to the store.
        If nodes_df is provided, it must contain columns: uuid, group_id, label, name, created_at, attributes.
        If edges_df is provided, it must contain columns: src_uuid, dst_uuid, type, event_ts, event_kind, attributes.
        The event_ts in edges_df is optional; if not provided, default_ts will be used.
        """
        p = self._paths(group_id)
        if nodes_df is not None and len(nodes_df):
            self._append_parquet(p.nodes, nodes_df)
        if edges_df is not None and len(edges_df):
            edges_df_copy = edges_df.copy()
            if "event_ts" not in edges_df_copy:
                edges_df_copy["event_ts"] = pd.Timestamp(
                    default_ts or datetime.now(timezone.utc), tz="UTC"
                )
            self._append_parquet(p.edge_events, edges_df_copy)

    def nodes_df(self, group_id: str, subset_uuids: list[str] | None = None) -> Any:
        """
        Returns a DataFrame of nodes for the given group_id.
        If subset_uuids is provided, only those nodes are returned.
        """

        p = self._paths(group_id)
        if not p.nodes.exists():
            return pd.DataFrame(
                columns=["uuid", "group_id", "label", "name", "created_at", "attributes"]
            )
        nodes_df = pq.read_table(p.nodes).to_pandas()
        nodes_df = nodes_df[nodes_df["group_id"] == group_id].copy()
        if subset_uuids:
            nodes_df = nodes_df[nodes_df["uuid"].isin(subset_uuids)]
        nodes_df.sort_values(["uuid", "created_at"], inplace=True)
        nodes_df = nodes_df.drop_duplicates(subset=["uuid"], keep="last")
        return nodes_df.reset_index(drop=True)

    def _intervals(self, *, group_id: str) -> Any:
        """
        Returns a DataFrame of edges with valid_from and valid_to timestamps.
        The valid_to is NaT if the edge is still valid.
        """

        p = self._paths(group_id)
        if not p.edge_events.exists():
            return pd.DataFrame(columns=["src", "dst", "type", "valid_from", "valid_to"])
        ev = pq.read_table(p.edge_events).to_pandas()
        ev = ev[ev["group_id"] == group_id].copy()
        if ev.empty:
            return pd.DataFrame(columns=["src", "dst", "type", "valid_from", "valid_to"])

        ev["event_ts"] = pd.to_datetime(ev["event_ts"], utc=True)
        ev.sort_values(["src_uuid", "dst_uuid", "type", "event_ts"], inplace=True)
        ev["next_ts"] = ev.groupby(["src_uuid", "dst_uuid", "type"], sort=False)["event_ts"].shift(
            -1
        )
        starts = ev[ev["event_kind"] == "open"].copy()
        starts.rename(
            columns={
                "src_uuid": "src",
                "dst_uuid": "dst",
                "event_ts": "valid_from",
                "next_ts": "valid_to",
            },
            inplace=True,
        )
        return starts[["src", "dst", "type", "valid_from", "valid_to"]]

    def edges_as_of(self, *, group_id: str, as_of: datetime) -> pd.DataFrame:
        """
        Returns edges that are valid as of the given timestamp.
        The timestamp is inclusive, meaning if an edge was valid at that time, it will be included.
        """

        iv = self._intervals(group_id=group_id)
        if iv.empty:
            return iv
        ts = pd.Timestamp(as_of, tz="UTC")
        mask = (iv["valid_from"] <= ts) & (iv["valid_to"].isna() | (iv["valid_to"] > ts))
        return iv.loc[mask, ["src", "dst", "type"]].reset_index(drop=True)

    def edges_in_window(self, *, group_id: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Returns edges that are valid in the given time window.
        The window is inclusive of start and exclusive of end.
        """

        iv = self._intervals(group_id=group_id)
        if iv.empty:
            return iv
        s = pd.Timestamp(start, tz="UTC")
        e = pd.Timestamp(end, tz="UTC")
        mask = (iv["valid_from"] < e) & (iv["valid_to"].isna() | (iv["valid_to"] > s))
        return iv.loc[mask, ["src", "dst", "type"]].reset_index(drop=True)
