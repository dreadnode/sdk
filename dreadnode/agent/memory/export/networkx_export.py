from datetime import datetime, timezone

import networkx as nx
from pandas import DataFrame

from dreadnode.agent.memory.backend.pandas_backend import PandasTemporalStore
from dreadnode.agent.memory.backend.steps import edges_at_step


def nx_snapshot_by_time(
    store: PandasTemporalStore,
    *,
    group_id: str,
    as_of: datetime | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    directed: bool = True,
) -> nx.Graph:
    if as_of is not None:
        edges = store.edges_as_of(group_id=group_id, as_of=as_of)
    elif start is not None and end is not None:
        edges = store.edges_in_window(group_id=group_id, start=start, end=end)
    else:
        edges = store.edges_as_of(group_id=group_id, as_of=datetime.now(timezone.utc))

    Gtype = nx.DiGraph if directed else nx.Graph
    G = nx.from_pandas_edgelist(
        edges, source="src", target="dst", edge_attr="type", create_using=Gtype()
    )
    return G


def nx_snapshot_by_step(
    store: PandasTemporalStore,
    steps_df: DataFrame,
    *,
    group_id: str,
    step_id: int,
    directed: bool = True,
) -> nx.Graph:
    edges = edges_at_step(store, steps_df, group_id, step_id)
    Gtype = nx.DiGraph if directed else nx.Graph
    return nx.from_pandas_edgelist(
        edges, source="src", target="dst", edge_attr="type", create_using=Gtype()
    )
