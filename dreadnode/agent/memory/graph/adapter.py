from datetime import datetime

import pandas as pd

from dreadnode.agent.memory.backend.pandas_backend import PandasTemporalStore
from dreadnode.agent.memory.graph.base import (
    Universe,
    coerce_edge_events_schema,
    coerce_nodes_schema,
)


class TypedGraphAdapter:
    def __init__(self, store: PandasTemporalStore, universe: Universe):
        self.store = store
        self.universe = universe

    def ingest(
        self,
        *,
        group_id: str,
        nodes_df: pd.DataFrame | None = None,
        edges_df: pd.DataFrame | None = None,
        default_ts: datetime | None = None,
    ) -> None:
        nodes_norm = None
        if nodes_df is not None and len(nodes_df):
            nd = nodes_df.copy()
            if self.universe.label_aliases and "label" in nd:
                nd["label"] = nd["label"].map(lambda x: self.universe.label_aliases.get(x, x))
            nodes_norm = coerce_nodes_schema(nd, allowed_labels=self.universe.labels)
            self.store.add_entities_df(group_id=group_id, nodes_df=nodes_norm)

        if edges_df is not None and len(edges_df):
            current_nodes = self.store.nodes_df(group_id)
            if nodes_norm is not None and len(nodes_norm):
                current_nodes = pd.concat([current_nodes, nodes_norm], ignore_index=True)
                current_nodes.sort_values(["uuid", "created_at"], inplace=True)
                current_nodes = current_nodes.drop_duplicates(subset=["uuid"], keep="last")
            edges_norm = coerce_edge_events_schema(
                edges_df, current_nodes, allowed_map=self.universe.allowed
            )
            self.store.add_entities_df(
                group_id=group_id, edges_df=edges_norm, default_ts=default_ts
            )
