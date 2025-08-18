from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from dreadnode.agent.memory.backend.pandas_backend import PandasTemporalStore

# ------------ Utilities ------------


def _edges_with_labels(
    store: PandasTemporalStore, group_id: str, edges: pd.DataFrame
) -> pd.DataFrame:
    """Annotate edges with src/dst labels and names from nodes."""
    if edges.empty:
        return edges.assign(
            src_label=pd.Series(dtype="string"),
            dst_label=pd.Series(dtype="string"),
            src_name=pd.Series(dtype="string"),
            dst_name=pd.Series(dtype="string"),
        )
    nodes = store.nodes_df(group_id)
    info = nodes[["uuid", "label", "name"]].copy()
    info.columns = ["uuid", "label", "name"]
    e = edges.merge(info, left_on="src", right_on="uuid", how="left")
    e.rename(columns={"label": "src_label", "name": "src_name"}, inplace=True)
    e.drop(columns=["uuid"], inplace=True)
    e = e.merge(info, left_on="dst", right_on="uuid", how="left")
    e.rename(columns={"label": "dst_label", "name": "dst_name"}, inplace=True)
    e.drop(columns=["uuid"], inplace=True)
    # enforce string dtype for clean joins/printing
    for c in ("src", "dst", "type", "src_label", "dst_label", "src_name", "dst_name"):
        e[c] = e[c].astype("string")
    return e


def _subset(e: pd.DataFrame, src_label: str, edge_type: str, dst_label: str) -> pd.DataFrame:
    if e.empty:
        return e
    m = (e["src_label"] == src_label) & (e["type"] == edge_type) & (e["dst_label"] == dst_label)
    out = e.loc[m, ["src", "src_name", "dst", "dst_name", "type"]].copy()
    out.rename(columns={"src": "src_uuid", "dst": "dst_uuid"}, inplace=True)
    return out.reset_index(drop=True)


# ------------ Findings (as-of OR window) ------------


def finding_user_has_host(
    store: PandasTemporalStore,
    group_id: str,
    *,
    as_of: datetime | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Active User →Has→ Host relations (sessions/access)"""
    edges = (
        store.edges_as_of(group_id, as_of or datetime.now(timezone.utc))
        if as_of
        else store.edges_in_window(group_id, start, end)
    )
    e = _edges_with_labels(store, group_id, edges)
    df = _subset(e, "User", "Has", "Host")
    df.columns = ["user_uuid", "user_name", "host_uuid", "host_name", "edge_type"]
    return df


def finding_user_has_credential(
    store: PandasTemporalStore,
    group_id: str,
    *,
    as_of: datetime | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Active User →Has→ Credential relations"""
    edges = (
        store.edges_as_of(group_id, as_of or datetime.now(timezone.utc))
        if as_of
        else store.edges_in_window(group_id, start, end)
    )
    e = _edges_with_labels(store, group_id, edges)
    df = _subset(e, "User", "Has", "Credential")
    df.columns = ["user_uuid", "user_name", "credential_uuid", "credential_name", "edge_type"]
    return df


def finding_credential_has_hash(
    store: PandasTemporalStore,
    group_id: str,
    *,
    as_of: datetime | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Credential →Has→ Hash linkages"""
    edges = (
        store.edges_as_of(group_id, as_of or datetime.now(timezone.utc))
        if as_of
        else store.edges_in_window(group_id, start, end)
    )
    e = _edges_with_labels(store, group_id, edges)
    df = _subset(e, "Credential", "Has", "Hash")
    df.columns = ["credential_uuid", "credential_name", "hash_uuid", "hash_name", "edge_type"]
    return df


def finding_hosts_with_weaknesses(
    store: PandasTemporalStore,
    group_id: str,
    *,
    as_of: datetime | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Host →Has→ Weakness (direct) OR Host→Has→Service and Service→Has→Weakness (via service)"""
    edges = (
        store.edges_as_of(group_id, as_of or datetime.now(timezone.utc))
        if as_of
        else store.edges_in_window(group_id, start, end)
    )
    e = _edges_with_labels(store, group_id, edges)

    # direct: Host Has Weakness
    direct = _subset(e, "Host", "Has", "Weakness")
    direct.columns = ["host_uuid", "host_name", "weakness_uuid", "weakness_name", "edge_type"]
    direct["via_service"] = False

    # via service: Host Has Service JOIN Service Has Weakness
    h_s = _subset(e, "Host", "Has", "Service")
    s_w = _subset(e, "Service", "Has", "Weakness")
    if not h_s.empty and not s_w.empty:
        j = h_s.merge(s_w, left_on="dst_uuid", right_on="src_uuid", suffixes=("_hs", "_sw"))
        via = j[["src_uuid_hs", "src_name_hs", "dst_uuid_sw", "dst_name_sw"]].copy()
        via.columns = ["host_uuid", "host_name", "weakness_uuid", "weakness_name"]
        via["edge_type"] = "Has"  # semantic: host is exposed via service weakness
        via["via_service"] = True
    else:
        via = pd.DataFrame(
            columns=[
                "host_uuid",
                "host_name",
                "weakness_uuid",
                "weakness_name",
                "edge_type",
                "via_service",
            ]
        )

    out = pd.concat([direct, via], ignore_index=True)
    return out.drop_duplicates().reset_index(drop=True)


def finding_weakness_impacts(
    store: PandasTemporalStore,
    group_id: str,
    *,
    as_of: datetime | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """Weakness →LedTo→ {Host, Service, User, Credential, Hash, Share}"""
    edges = (
        store.edges_as_of(group_id, as_of or datetime.now(timezone.utc))
        if as_of
        else store.edges_in_window(group_id, start, end)
    )
    e = _edges_with_labels(store, group_id, edges)
    m = (e["src_label"] == "Weakness") & (e["type"] == "LedTo")
    df = e.loc[m, ["src", "src_name", "dst", "dst_name", "dst_label"]].copy()
    df.columns = ["weakness_uuid", "weakness_name", "impact_uuid", "impact_name", "impact_label"]
    return df.reset_index(drop=True)


def finding_two_hop_weakness_to_host_via_user(
    store: PandasTemporalStore,
    group_id: str,
    *,
    as_of: datetime | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """
    2-hop path: Weakness --LedTo--> User --Has--> Host
    This respects your allowed types and is useful to see indirect host exposure driven by a Weakness.
    """
    edges = (
        store.edges_as_of(group_id, as_of or datetime.now(timezone.utc))
        if as_of
        else store.edges_in_window(group_id, start, end)
    )
    if edges.empty:
        return pd.DataFrame(
            columns=[
                "weakness_uuid",
                "weakness_name",
                "user_uuid",
                "user_name",
                "host_uuid",
                "host_name",
            ]
        )

    e = _edges_with_labels(store, group_id, edges)

    w_u = e[(e["src_label"] == "Weakness") & (e["type"] == "LedTo") & (e["dst_label"] == "User")]
    u_h = e[(e["src_label"] == "User") & (e["type"] == "Has") & (e["dst_label"] == "Host")]
    if w_u.empty or u_h.empty:
        return pd.DataFrame(
            columns=[
                "weakness_uuid",
                "weakness_name",
                "user_uuid",
                "user_name",
                "host_uuid",
                "host_name",
            ]
        )

    j = w_u.merge(u_h, left_on="dst", right_on="src", suffixes=("_wu", "_uh"))
    out = j[["src_wu", "src_name_wu", "dst_wu", "dst_name_wu", "dst_uh", "dst_name_uh"]].copy()
    out.columns = [
        "weakness_uuid",
        "weakness_name",
        "user_uuid",
        "user_name",
        "host_uuid",
        "host_name",
    ]
    return out.drop_duplicates().reset_index(drop=True)


def finding_user_has_share(
    store: PandasTemporalStore,
    group_id: str,
    *,
    as_of: datetime | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """User →Has→ Share (potential data exposure)"""
    edges = (
        store.edges_as_of(group_id, as_of or datetime.now(timezone.utc))
        if as_of
        else store.edges_in_window(group_id, start, end)
    )
    e = _edges_with_labels(store, group_id, edges)
    df = _subset(e, "User", "Has", "Share")
    df.columns = ["user_uuid", "user_name", "share_uuid", "share_name", "edge_type"]
    return df


# ------------ Step-aware wrappers ------------


def _window_for_step(steps_df: pd.DataFrame, step_id: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    row = steps_df.loc[steps_df["step_id"] == step_id]
    if row.empty:
        raise KeyError(f"step_id {step_id} not found")
    return row["start_ts"].iloc[0], row["end_ts"].iloc[0]


def run_all_findings_as_of(
    store: PandasTemporalStore, group_id: str, as_of: datetime
) -> dict[str, pd.DataFrame]:
    return {
        "user_has_host": finding_user_has_host(store, group_id, as_of=as_of),
        "user_has_credential": finding_user_has_credential(store, group_id, as_of=as_of),
        "credential_has_hash": finding_credential_has_hash(store, group_id, as_of=as_of),
        "hosts_with_weaknesses": finding_hosts_with_weaknesses(store, group_id, as_of=as_of),
        "weakness_impacts": finding_weakness_impacts(store, group_id, as_of=as_of),
        "weakness_to_host_via_user": finding_two_hop_weakness_to_host_via_user(
            store, group_id, as_of=as_of
        ),
        "user_has_share": finding_user_has_share(store, group_id, as_of=as_of),
    }


def run_all_findings_window(
    store: PandasTemporalStore, group_id: str, start: datetime, end: datetime
) -> dict[str, pd.DataFrame]:
    return {
        "user_has_host": finding_user_has_host(store, group_id, start=start, end=end),
        "user_has_credential": finding_user_has_credential(store, group_id, start=start, end=end),
        "credential_has_hash": finding_credential_has_hash(store, group_id, start=start, end=end),
        "hosts_with_weaknesses": finding_hosts_with_weaknesses(
            store, group_id, start=start, end=end
        ),
        "weakness_impacts": finding_weakness_impacts(store, group_id, start=start, end=end),
        "weakness_to_host_via_user": finding_two_hop_weakness_to_host_via_user(
            store, group_id, start=start, end=end
        ),
        "user_has_share": finding_user_has_share(store, group_id, start=start, end=end),
    }


def run_all_findings_step(
    store: PandasTemporalStore, group_id: str, steps_df: pd.DataFrame, step_id: int
) -> dict[str, pd.DataFrame]:
    s, e = _window_for_step(steps_df, step_id)
    return run_all_findings_window(store, group_id, start=s, end=e)


# ------------ Reporting helpers ------------


@dataclass
class ReportArtifacts:
    written_csvs: list[Path]
    markdown: str


def write_findings_csv(findings: dict[str, pd.DataFrame], out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, df in findings.items():
        p = out_dir / f"{name}.csv"
        df.to_csv(p, index=False)
        written.append(p)
    return written


def findings_to_markdown(findings: dict[str, pd.DataFrame], max_rows: int = 20) -> str:
    lines: list[str] = ["# Findings Summary\n"]
    for name, df in findings.items():
        lines.append(f"## {name.replace('_', ' ').title()}  \nRows: {len(df)}\n")
        if df.empty:
            lines.append("_No rows_\n")
            continue
        head = df.head(max_rows)
        lines.append(head.to_markdown(index=False))
        if len(df) > max_rows:
            lines.append(f"\n… and {len(df) - max_rows} more rows.\n")
        lines.append("\n")
    return "\n".join(lines)


def save_report(
    findings: dict[str, pd.DataFrame], out_dir: str | Path, max_rows: int = 20
) -> ReportArtifacts:
    csvs = write_findings_csv(findings, out_dir)
    md = findings_to_markdown(findings, max_rows=max_rows)
    (Path(out_dir) / "report.md").write_text(md, encoding="utf-8")
    return ReportArtifacts(written_csvs=csvs, markdown=md)
