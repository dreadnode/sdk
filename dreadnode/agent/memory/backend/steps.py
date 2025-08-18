from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dreadnode.agent.memory.backend.pandas_backend import PandasTemporalStore

STEPS_SCHEMA = pa.schema(
    [
        pa.field("group_id", pa.string()),
        pa.field("step_id", pa.int64()),
        pa.field("start_ts", pa.timestamp("us", tz="UTC")),
        pa.field("end_ts", pa.timestamp("us", tz="UTC")),
        pa.field("center_ts", pa.timestamp("us", tz="UTC")),
        pa.field("label", pa.string()),
    ]
)


def _steps_path(root: Path, group_id: str) -> Path:
    p = root / (group_id or "default") / "steps.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def build_steps_fixed(
    root: str | Path,
    group_id: str,
    *,
    freq: str = "1H",
    start: datetime | None = None,
    end: datetime | None = None,
    label_prefix: str = "",
) -> pd.DataFrame:
    root = Path(root)
    ev_path = root / (group_id or "default") / "edge_events.parquet"
    if (start is None or end is None) and ev_path.exists():
        ev = pq.read_table(ev_path, columns=["event_ts"]).to_pandas()
        if not ev.empty:
            s_min = pd.to_datetime(ev["event_ts"], utc=True).min()
            s_max = pd.to_datetime(ev["event_ts"], utc=True).max()
            start = start or s_min.floor(freq)
            end = end or (s_max.ceil(freq) + pd.Timedelta(freq))
    if start is None or end is None:
        raise ValueError("Provide start/end or ingest edge events first.")

    rng = pd.date_range(start=start, end=end, freq=freq, inclusive="left", tz="UTC")
    df = pd.DataFrame(
        {
            "group_id": group_id,
            "step_id": range(len(rng)),
            "start_ts": rng,
            "end_ts": rng + pd.Timedelta(freq),
        }
    )
    df["center_ts"] = df["start_ts"] + (df["end_ts"] - df["start_ts"]) / 2
    df["label"] = [f"{label_prefix}{i}" for i in df["step_id"]]

    path = _steps_path(root, group_id)
    pq.write_table(
        pa.Table.from_pandas(df, preserve_index=False, schema=STEPS_SCHEMA),
        path,
        existing_data_behavior="overwrite",
    )
    return df


def load_steps(root: str | Path, group_id: str) -> pd.DataFrame:
    path = _steps_path(Path(root), group_id)
    if not path.exists():
        return pd.DataFrame(
            columns=["group_id", "step_id", "start_ts", "end_ts", "center_ts", "label"]
        )
    return pq.read_table(path).to_pandas()


def time_to_step(steps_df: pd.DataFrame, ts: datetime) -> int | None:
    t = pd.Timestamp(ts, tz="UTC")
    m = (steps_df["start_ts"] <= t) & (t < steps_df["end_ts"])
    if not m.any():
        return None
    return int(steps_df.loc[m, "step_id"].iloc[0])


def step_to_time(steps_df: pd.DataFrame, step_id: int, which: str = "center") -> pd.Timestamp:
    row = steps_df.loc[steps_df["step_id"] == step_id]
    if row.empty:
        raise KeyError(f"step_id {step_id} not found")
    return pd.Timestamp(row[f"{which}_ts"].iloc[0], tz="UTC")


# ----- snapshot helpers -----
def edges_at_step(
    store: PandasTemporalStore, steps_df: pd.DataFrame, group_id: str, step_id: int
) -> pd.DataFrame:
    row = steps_df.loc[steps_df["step_id"] == step_id]
    if row.empty:
        return pd.DataFrame(columns=["src", "dst", "type"])
    return store.edges_in_window(
        group_id=group_id, start=row["start_ts"].iloc[0], end=row["end_ts"].iloc[0]
    )


# (optional) delta-based multi-step acceleration
def step_deltas_from_events(
    root: str | Path, group_id: str, steps_df: pd.DataFrame
) -> pd.DataFrame:
    root = Path(root)
    ev_path = root / (group_id or "default") / "edge_events.parquet"
    if not ev_path.exists():
        return pd.DataFrame(columns=["src", "dst", "type", "step_id", "delta"])
    ev = pq.read_table(ev_path).to_pandas()
    ev["event_ts"] = pd.to_datetime(ev["event_ts"], utc=True)

    bounds = steps_df[["step_id", "start_ts"]].rename(columns={"start_ts": "t"}).sort_values("t")
    ev_sorted = ev.sort_values("event_ts")
    merged = pd.merge_asof(
        ev_sorted, bounds, left_on="event_ts", right_on="t", direction="backward"
    )
    out = merged[["src_uuid", "dst_uuid", "type", "step_id", "event_kind"]].copy()
    out["delta"] = out["event_kind"].map({"open": +1, "close": -1}).fillna(+1)
    out.rename(columns={"src_uuid": "src", "dst_uuid": "dst"}, inplace=True)
    return out[["src", "dst", "type", "step_id", "delta"]]


def edges_active_matrix(step_deltas: pd.DataFrame) -> pd.DataFrame:
    sd = step_deltas.sort_values(["src", "dst", "type", "step_id"])
    sd["cum"] = sd.groupby(["src", "dst", "type"], sort=False)["delta"].cumsum()
    active = sd[sd["cum"] > 0][["src", "dst", "type", "step_id"]].copy()
    active["active"] = 1
    return active


def edges_at_step_from_active(active_long: pd.DataFrame, step_id: int) -> pd.DataFrame:
    m = active_long["step_id"] == step_id
    return active_long.loc[m, ["src", "dst", "type"]].reset_index(drop=True)
