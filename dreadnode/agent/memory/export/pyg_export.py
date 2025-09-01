import json
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd
import torch
from torch_geometric.data import HeteroData

from dreadnode.agent.memory import PandasTemporalStore


@dataclass
class HeteroExportResult:
    data: HeteroData
    node_id_maps: dict[str, dict[str, int]]  # node_type -> {uuid -> idx}
    edge_counts: dict[tuple[str, str, str], int]  # (src_type, rel_type, dst_type) -> count


def export_pyg_heterodata(
    store: PandasTemporalStore,
    group_id: str,
    *,
    as_of: datetime | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    include_isolated: bool = False,
    feature_key_by_type: dict[str, str] | None = None,
) -> HeteroExportResult:
    if as_of is not None:
        e_df = store.edges_as_of(group_id=group_id, as_of=as_of)
    elif start is not None and end is not None:
        e_df = store.edges_in_window(group_id=group_id, start=start, end=end)
    else:
        e_df = store.edges_as_of(group_id=group_id, as_of=datetime.now(timezone.utc))

    e_df = e_df.astype({"src": "string", "dst": "string", "type": "string"})
    used = set(e_df["src"]).union(set(e_df["dst"]))
    n_df = store.nodes_df(group_id=group_id, subset_uuids=None if include_isolated else list(used))
    if n_df.empty:
        return HeteroExportResult(HeteroData(), {}, {})

    def _to_dict(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, str) and x:
            try:
                return json.loads(x)
            except Exception:
                return {}
        return {}

    n_df["attributes"] = n_df.get("attributes", "{}").apply(_to_dict)
    n_df["label"] = n_df["label"].astype("string")
    n_df["uuid"] = n_df["uuid"].astype("string")

    node_id_maps: dict[str, dict[str, int]] = {}
    type_to_nodes: dict[str, list[str]] = {}
    for t, sub in n_df.groupby("label", sort=False):
        uuids = sub["uuid"].tolist()
        type_to_nodes[t] = uuids
        node_id_maps[t] = {u: i for i, u in enumerate(uuids)}

    data = HeteroData()
    edge_counts: dict[tuple[str, str, str], int] = {}
    feature_key_by_type = feature_key_by_type or {}

    for t, uuids in type_to_nodes.items():
        data[t].num_nodes = len(uuids)
        feat_key = feature_key_by_type.get(t)
        if feat_key:
            feats = []
            ok = True
            sub = n_df[n_df["label"] == t]
            for _, row in sub.iterrows():
                val = row["attributes"].get(feat_key)
                if isinstance(val, (list, tuple)) and all(isinstance(v, (int, float)) for v in val):
                    feats.append(val)
                else:
                    ok = False
                    break
            if ok and feats:
                data[t].x = torch.tensor(feats, dtype=torch.float32)

    uuid_to_type = dict(zip(n_df["uuid"], n_df["label"], strict=False))
    rows = []
    for src, dst, rtype in e_df[["src", "dst", "type"]].itertuples(index=False):
        s_type = uuid_to_type.get(src)
        d_type = uuid_to_type.get(dst)
        if s_type is None or d_type is None:
            continue
        rows.append((s_type, rtype, d_type, src, dst))

    if not rows:
        return HeteroExportResult(data, node_id_maps, edge_counts)

    df_edges = pd.DataFrame(rows, columns=["s_type", "r_type", "d_type", "src", "dst"])
    for (s_type, r_type, d_type), grp in df_edges.groupby(
        ["s_type", "r_type", "d_type"], sort=False
    ):
        s_map = node_id_maps[s_type]
        d_map = node_id_maps[d_type]
        src_idx = [s_map[u] for u in grp["src"]]
        dst_idx = [d_map[u] for u in grp["dst"]]
        edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
        data[(s_type, r_type, d_type)].edge_index = edge_index
        edge_counts[(s_type, r_type, d_type)] = edge_index.size(1)

    return HeteroExportResult(data=data, node_id_maps=node_id_maps, edge_counts=edge_counts)
