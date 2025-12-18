import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypedDict

import pandas as pd

from dreadnode.agent.graph.adapter import TypedGraphAdapter
from dreadnode.agent.graph.backend import PandasTemporalStore
from dreadnode.agent.graph.export import nx_snapshot_by_time

logger = logging.getLogger(__name__)
pd.options.mode.copy_on_write = True


# ---- public response types (same as your tool API) ----
class SuccessResponse(TypedDict):
    message: str


class ErrorResponse(TypedDict):
    error: str


class NodeResult(TypedDict):
    uuid: str
    name: str
    summary: str
    labels: list[str]
    group_id: str
    created_at: str
    attributes: dict[str, Any]


class NodeSearchResponse(TypedDict):
    message: str
    nodes: list[NodeResult]


class FactSearchResponse(TypedDict):
    message: str
    facts: list[dict[str, Any]]


def _to_iso(dt) -> str:
    if pd.isna(dt) or dt is None:
        return datetime.now(timezone.utc).isoformat()
    ts = pd.to_datetime(dt, utc=True)
    return ts.isoformat()


def _to_attrs(x) -> dict[str, Any]:
    if isinstance(x, dict):
        return x
    if isinstance(x, str) and x:
        try:
            return json.loads(x)
        except Exception:
            return {"_raw": x}
    return {}


def serialize_nodes_df(nodes_df: pd.DataFrame, group_id: str) -> list[NodeResult]:
    out: list[NodeResult] = []
    for r in nodes_df.itertuples(index=False):
        attrs = getattr(r, "attributes", "{}")
        out.append(
            NodeResult(
                uuid=str(r.uuid),
                name=str(getattr(r, "name", "") or ""),
                summary="",  # fill if you compute one
                labels=[str(r.label)],
                group_id=group_id,
                created_at=_to_iso(getattr(r, "created_at", None)),
                attributes=_to_attrs(attrs),
            )
        )
    return out


@dataclass
class MemoryConfig:
    root: str = os.environ.get("LAKE_ROOT", "./lake_pandas")
    default_group_id: str | None = os.environ.get("GROUP_ID")
    destroy_on_start: bool = bool(int(os.environ.get("DESTROY_ON_START", "0")))


def _wipe_all_parquet(root: str):
    import pathlib
    import shutil

    p = pathlib.Path(root)
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def _format_fact_result(edge_row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    src = edge_row.get("src") or edge_row.get("src_uuid")
    dst = edge_row.get("dst") or edge_row.get("dst_uuid")
    out["src_uuid"] = str(src or "")
    out["dst_uuid"] = str(dst or "")
    out["type"] = str(edge_row.get("type", ""))
    if "event_ts" in edge_row:
        out["event_ts"] = pd.to_datetime(edge_row["event_ts"], utc=True).isoformat()
        out["event_kind"] = str(edge_row.get("event_kind", "open"))
    if "valid_from" in edge_row or "valid_to" in edge_row:
        if edge_row.get("valid_from") is not None:
            out["valid_from"] = pd.to_datetime(edge_row["valid_from"], utc=True).isoformat()
        if edge_row.get("valid_to") is not None:
            out["valid_to"] = pd.to_datetime(edge_row["valid_to"], utc=True).isoformat()
    attrs = edge_row.get("attributes", {})
    if isinstance(attrs, str):
        try:
            attrs = json.loads(attrs)
        except Exception:
            attrs = {"_raw": attrs}
    out["attributes"] = attrs if isinstance(attrs, dict) else {}
    return out


class MemoryService:
    def __init__(
        self, config: MemoryConfig, store: PandasTemporalStore, adapter: TypedGraphAdapter
    ):
        self.config = config
        self.store = store
        self.adapter = adapter
        self._queues: dict[str, asyncio.Queue] = {}
        self._workers: dict[str, asyncio.Task] = {}

    @classmethod
    async def create(cls, config: MemoryConfig | None = None) -> "MemoryService":
        cfg = config or MemoryConfig()
        if cfg.destroy_on_start:
            _wipe_all_parquet(cfg.root)
        store = PandasTemporalStore(root=cfg.root)
        adapter = TypedGraphAdapter(store)  # uses CORE_UNIVERSE by default
        logger.info("Temporal memory initialized (pandas store at %s)", cfg.root)
        return cls(cfg, store, adapter)

    async def shutdown(self):
        # cancel worker tasks cleanly
        for gid, t in list(self._workers.items()):
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("Worker shutdown error for %s", gid)
        self._workers.clear()

    # ---------- internal queue worker ----------
    async def _process_episode_queue(self, group_id: str):
        logger.info("Starting episode queue worker for group_id: %s", group_id)
        q = self._queues[group_id]
        try:
            while True:
                fn = await q.get()
                try:
                    await fn()
                except Exception as e:
                    logger.exception("Error processing episode for %s: %s", group_id, e)
                finally:
                    q.task_done()
        except asyncio.CancelledError:
            logger.info("Episode queue worker for %s cancelled", group_id)
        finally:
            self._workers.pop(group_id, None)
            logger.info("Stopped episode queue worker for %s", group_id)

    def _ensure_worker(self, group_id: str):
        if group_id not in self._queues:
            self._queues[group_id] = asyncio.Queue()
        if group_id not in self._workers:
            self._workers[group_id] = asyncio.create_task(self._process_episode_queue(group_id))

    # ---------- public API (same semantics as before) ----------
    async def add_memory(
        self,
        name: str,
        episode_body: str,
        group_id: str | None = None,
        source: str = "text",
        source_description: str = "",
        uuid: str | None = None,
    ) -> SuccessResponse | ErrorResponse:
        gid = (group_id or self.config.default_group_id) or "default"
        now = datetime.now(timezone.utc)

        async def process_episode():
            logger.info("Processing episode '%s' (group=%s, source=%s)", name, gid, source)
            s = source.lower().strip()
            try:
                if s == "json":
                    payload = json.loads(episode_body)
                    nodes = payload.get("nodes", [])
                    edges = payload.get("edges", [])
                    nodes_df = pd.DataFrame(nodes) if nodes else None
                    edges_df = pd.DataFrame(edges) if edges else None
                    self.adapter.ingest(
                        group_id=gid, nodes_df=nodes_df, edges_df=edges_df, default_ts=now
                    )
                else:
                    logger.info("Text/message episode stored (no IE). desc=%s", source_description)
            except Exception as e:
                logger.exception("Episode '%s' failed: %s", name, e)

        self._ensure_worker(gid)
        await self._queues[gid].put(process_episode)
        pos = self._queues[gid].qsize()
        return SuccessResponse(message=f"Episode '{name}' queued for processing (position: {pos})")

    async def search_memory_nodes(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_nodes: int = 10,
        center_node_uuid: str | None = None,
        entity: str = "",
    ) -> NodeSearchResponse | ErrorResponse:
        try:
            gids = (
                group_ids
                or ([self.config.default_group_id] if self.config.default_group_id else [])
                or ["default"]
            )
            asof = datetime.now(timezone.utc)
            results: list[NodeResult] = []

            for gid in gids:
                ndf = self.store.nodes_df(gid)
                if ndf.empty:
                    continue

                hay = (ndf["name"].fillna("") + " " + ndf["attributes"].fillna("")).str.lower()
                mask = (
                    hay.str.contains(query.lower(), na=False)
                    if query
                    else pd.Series([True] * len(ndf))
                )
                if entity:
                    mask = mask & (ndf["label"] == entity)
                sub = ndf[mask].copy()

                if center_node_uuid:
                    try:
                        G = nx_snapshot_by_time(self.store, group_id=gid, as_of=asof, directed=True)
                        from collections import deque

                        dist = {}
                        if center_node_uuid in G:
                            q = deque([(center_node_uuid, 0)])
                            seen = {center_node_uuid}
                            while q:
                                u, d = q.popleft()
                                dist[u] = d
                                for v in G.neighbors(u):
                                    if v not in seen:
                                        seen.add(v)
                                        q.append((v, d + 1))
                            sub["__dist"] = sub["uuid"].map(lambda u: dist.get(u, 10**9))
                            sub.sort_values(["__dist", "name"], inplace=True)
                        else:
                            sub["__dist"] = 10**9
                    except Exception:
                        sub["__dist"] = 10**9
                else:
                    sub["__dist"] = 0

                for r in sub.head(max_nodes).itertuples(index=False):
                    try:
                        attrs = r.attributes
                        if isinstance(attrs, str):
                            attrs = json.loads(attrs) if attrs else {}
                    except Exception:
                        attrs = {}
                    results.append(
                        {
                            "uuid": str(r.uuid),
                            "name": str(r.name or ""),
                            "summary": "",
                            "labels": [str(r.label)],
                            "group_id": gid,
                            "created_at": pd.to_datetime(r.created_at, utc=True).isoformat()
                            if "created_at" in r._fields
                            else datetime.now(timezone.utc).isoformat(),
                            "attributes": attrs if isinstance(attrs, dict) else {},
                        }
                    )

            return NodeSearchResponse(
                message="Nodes retrieved successfully", nodes=results[:max_nodes]
            )
        except Exception as e:
            logger.exception("search_memory_nodes error: %s", e)
            return ErrorResponse(error=str(e))

    async def search_memory_facts(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_facts: int = 10,
        center_node_uuid: str | None = None,
    ) -> FactSearchResponse | ErrorResponse:
        try:
            gids = (
                group_ids
                or ([self.config.default_group_id] if self.config.default_group_id else [])
                or ["default"]
            )
            asof = datetime.now(timezone.utc)
            facts_out: list[dict[str, Any]] = []

            for gid in gids:
                edges = self.store.edges_as_of(group_id=gid, as_of=asof)
                if edges.empty:
                    continue
                nodes = self.store.nodes_df(gid)[["uuid", "label", "name", "attributes"]]
                e = edges.merge(
                    nodes.add_prefix("src_"), left_on="src", right_on="src_uuid", how="left"
                )
                e = e.merge(
                    nodes.add_prefix("dst_"), left_on="dst", right_on="dst_uuid", how="left"
                )

                if center_node_uuid:
                    e = e[(e["src"] == center_node_uuid) | (e["dst"] == center_node_uuid)]

                if query:
                    q = query.lower()
                    cols = [
                        e["type"].astype(str).str.lower(),
                        e["src_label"].astype(str).str.lower(),
                        e["src_name"].astype(str).str.lower(),
                        e["dst_label"].astype(str).str.lower(),
                        e["dst_name"].astype(str).str.lower(),
                        e["src_attributes"].astype(str).str.lower(),
                        e["dst_attributes"].astype(str).str.lower(),
                    ]
                    hay = cols[0]
                    for c in cols[1:]:
                        hay = hay.str.cat(c, sep=" ")
                    e = e[hay.str.contains(q, na=False)]

                facts_out.extend(
                    [
                        _format_fact_result(
                            {
                                "src_uuid": row["src"],
                                "dst_uuid": row["dst"],
                                "type": row["type"],
                                "attributes": {
                                    "src": {
                                        "uuid": row["src"],
                                        "label": row.get("src_label"),
                                        "name": row.get("src_name"),
                                    },
                                    "dst": {
                                        "uuid": row["dst"],
                                        "label": row.get("dst_label"),
                                        "name": row.get("dst_name"),
                                    },
                                },
                            }
                        )
                        for row in e.head(max_facts).to_dict("records")
                    ]
                )

            return FactSearchResponse(
                message="Facts retrieved successfully", facts=facts_out[:max_facts]
            )
        except Exception as e:
            logger.exception("search_memory_facts error: %s", e)
            return ErrorResponse(error=str(e))
