"""
Minimal Streamlit dashboard for Kairos explanation artifacts.

Run from the /cadets directory:
    streamlit run reporting/streamlit_dashboard.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytz
import streamlit as st
import plotly.graph_objects as go

CAD_DIR = Path(__file__).resolve().parents[1]
if str(CAD_DIR) not in sys.path:
    sys.path.insert(0, str(CAD_DIR))

import config  # type: ignore

EST = pytz.timezone("US/Eastern")
EXPLANATION_DIR = CAD_DIR / "artifact" / "explanations"
DEFAULT_PATTERN = "*_explanations.json"
MAX_ROWS = 15


@st.cache_data(show_spinner=False)
def load_mapping(path: Optional[Path]) -> Dict[int, str]:
    if not path or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(k): v for k, v in data.items()}


@st.cache_data(show_spinner=False)
def load_payload(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def to_local_time(value: Optional[int]) -> str:
    if value is None:
        return "—"
    seconds, nanos = divmod(int(value), 1_000_000_000)
    dt = pytz.UTC.localize(datetime.utcfromtimestamp(seconds)).astimezone(EST)
    return f"{dt.strftime('%Y-%m-%d %H:%M:%S')}.{str(nanos).zfill(9)}"


def resolve_label(node_id: Optional[int], mapping: Dict[int, str]) -> str:
    if node_id is None:
        return "Unknown"
    return mapping.get(int(node_id), f"Node {node_id}")


def summarise_graph_edges(payload: Dict[str, object], mapping: Dict[int, str]) -> List[Dict[str, object]]:
    aggregate = payload.get("graphmask", {}).get("aggregate")
    if not isinstance(aggregate, list):
        return []
    rows: List[Dict[str, object]] = []
    for item in aggregate:
        src = resolve_label(item.get("src"), mapping)
        dst = resolve_label(item.get("dst"), mapping)
        relation = item.get("relation", "?")
        weight = float(item.get("weight", 0.0))
        count = int(item.get("count", 0))
        timestamps = item.get("timestamps", [])
        first_seen = to_local_time(timestamps[0]) if timestamps else "—"
        last_seen = to_local_time(timestamps[-1]) if timestamps else "—"
        rows.append(
            {
                "relation": relation,
                "source": src,
                "destination": dst,
                "weight": f"{weight:.3f}",
                "count": str(count),
                "first_seen": first_seen,
                "last_seen": last_seen,
            }
        )
    rows.sort(key=lambda r: float(r["weight"]), reverse=True)
    return rows


def summarise_nodes(payload: Dict[str, object], mapping: Dict[int, str], threshold: float) -> List[Dict[str, object]]:
    nodes = payload.get("nodes", [])
    if not isinstance(nodes, list):
        return []
    rows: List[Dict[str, object]] = []
    for node in nodes:
        label = resolve_label(node.get("node_id"), mapping)
        avg_loss = float(node.get("avg_score", node.get("score", 0.0)))
        total_loss = float(node.get("score", 0.0))
        event_count = int(node.get("event_count", 0))
        peak_loss = float(node.get("max_event_loss", 0.0))
        above = float(node.get("max_event_loss", 0.0)) >= threshold
        rows.append(
            {
                "label": label,
                "avg_loss": f"{avg_loss:.3f}",
                "peak_loss": f"{peak_loss:.3f}",
                "events": str(event_count),
                "total_loss": f"{total_loss:,.3f}",
                "above": above,
            }
        )
    rows.sort(key=lambda r: float(r["avg_loss"]), reverse=True)
    return rows


def summarise_node_detail(node: Dict[str, object], mapping: Dict[int, str]) -> Dict[str, List[Tuple[str, str]]]:
    def flatten_edges(source: List[Dict[str, object]]) -> List[Tuple[str, str]]:
        rows: List[Tuple[str, str]] = []
        for entry in source or []:
            relation = entry.get("relation", "?")
            src = resolve_label(entry.get("src"), mapping) if entry.get("src") is not None else "Unknown"
            dst = resolve_label(entry.get("dst"), mapping) if entry.get("dst") is not None else "Unknown"
            weight = float(entry.get("weight", 0.0))
            rows.append((f"{relation} {src} → {dst}", f"{weight:.3f}"))
        rows.sort(key=lambda r: float(r[1]), reverse=True)
        return rows

    gnn_edges: List[Tuple[str, str]] = []
    for event in node.get("gnn", []) or []:
        gnn_edges.extend(flatten_edges(event.get("top_edges", []))[:5])

    va_edges: List[Tuple[str, str]] = []
    for entry in node.get("va_tg", {}).get("events", []) or []:
        va_edges.extend(flatten_edges(entry.get("edges", []))[:5])
    va_aggregate = flatten_edges(node.get("va_tg", {}).get("aggregate", []))

    return {
        "gnn_edges": gnn_edges[:MAX_ROWS],
        "va_edges": va_edges[:MAX_ROWS],
        "va_aggregate": va_aggregate[:MAX_ROWS],
    }


def render_table(title: str, rows: List[Dict[str, object]], columns: List[str], empty_message: str = "No data.") -> None:
    st.markdown(f"#### {title}")
    if not rows:
        st.info(empty_message)
        return
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows[:MAX_ROWS]:
        cells = [str(row.get(column, "—")) for column in columns]
        body.append("| " + " | ".join(cells) + " |")
    st.markdown("\n".join([header, divider, *body]))
    if len(rows) > MAX_ROWS:
        st.caption(f"Showing first {MAX_ROWS} of {len(rows)} rows.")


def render_pairs(title: str, pairs: List[Tuple[str, str]]) -> None:
    st.markdown(f"##### {title}")
    if not pairs:
        st.info("No entries.")
        return
    header = "| Item | Weight |"
    divider = "| --- | ---: |"
    body = ["| " + label + " | " + weight + " |" for label, weight in pairs[:MAX_ROWS]]
    st.markdown("\n".join([header, divider, *body]))


def _relation_color(rel: str) -> str:
    palette = {
        "EVENT_WRITE": "#1f77b4",
        "EVENT_READ": "#2ca02c",
        "EVENT_CLOSE": "#9467bd",
        "EVENT_OPEN": "#8c564b",
        "EVENT_EXECUTE": "#d62728",
        "EVENT_SENDTO": "#ff7f0e",
        "EVENT_RECVFROM": "#17becf",
    }
    return palette.get(rel, "#7f7f7f")


def plot_top_edges_bar(graph_rows: List[Dict[str, object]], k: int = 10) -> go.Figure:
    top = graph_rows[:k]
    labels = [f"{r['relation']} {r['source']} → {r['destination']}" for r in top]
    weights = [float(r["weight"]) for r in top]
    colors = [_relation_color(r["relation"]) for r in top]
    hover = [
        f"<b>{r['relation']}</b><br>src: {r['source']}<br>dst: {r['destination']}<br>weight: {r['weight']}<br>count: {r['count']}<br>first: {r['first_seen']}<br>last: {r['last_seen']}"
        for r in top
    ]
    fig = go.Figure(
        go.Bar(
            x=weights,
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{customdata}",
            customdata=hover,
        )
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    fig.update_yaxes(automargin=True)
    return fig


def plot_node_scores_bar(node_rows: List[Dict[str, object]], threshold: float, k: int = 10) -> go.Figure:
    top = node_rows[:k]
    labels = [r["label"] for r in top]
    scores = [float(r["avg_loss"]) for r in top]
    colors = ["#d62728" if r.get("above") else "#2ca02c" for r in top]
    fig = go.Figure(
        go.Bar(x=scores, y=labels, orientation="h", marker_color=colors, hovertemplate="avg_loss=%{x:.3f}<extra></extra>")
    )
    fig.add_shape(
        type="line",
        x0=threshold,
        x1=threshold,
        y0=-0.5,
        y1=len(labels) - 0.5,
        line=dict(color="#7f7f7f", width=2, dash="dash"),
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    fig.update_yaxes(automargin=True)
    return fig


def plot_event_timeline(payload: Dict[str, object]) -> go.Figure:
    # Use all aggregate timestamps across edges; bin by minute.
    aggregate = payload.get("graphmask", {}).get("aggregate")
    if not isinstance(aggregate, list):
        return go.Figure()
    counts: Dict[str, int] = {}
    for entry in aggregate:
        for ts in entry.get("timestamps", []) or []:
            seconds, _ = divmod(int(ts), 1_000_000_000)
            dt = pytz.UTC.localize(datetime.utcfromtimestamp(seconds)).astimezone(EST)
            key = dt.strftime("%Y-%m-%d %H:%M")
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return go.Figure()
    items = sorted(counts.items())
    x = [k for k, _ in items]
    y = [v for _, v in items]
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers", line=dict(color="#1f77b4")))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    return fig


def parse_gpt_sections(md: Optional[str]) -> Dict[str, str]:
    if not md:
        return {}
    sections: Dict[str, str] = {}
    current_title: Optional[str] = None
    current_lines: List[str] = []
    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("## "):
            if current_title is not None:
                sections[current_title] = "\n".join(current_lines).strip()
            current_title = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_title is not None:
        sections[current_title] = "\n".join(current_lines).strip()
    return sections


def main() -> None:
    st.set_page_config(page_title="XAI-Kairos", layout="wide")
    st.title("XAI - Kairos SOC Dashboard")
    st.caption("Minimal analyst view for Kairos explanation artifacts.")

    mapping_default = Path(getattr(config, "NODE_MAPPING_JSON", EXPLANATION_DIR / "node_mapping.json"))
    mapping_path = Path(os.environ.get("KAIROS_NODE_MAPPING_JSON", mapping_default))
    node_map = load_mapping(mapping_path if mapping_path.exists() else None)

    files = sorted(EXPLANATION_DIR.glob(DEFAULT_PATTERN))
    if not files:
        st.warning("No explanation JSON files found under artifact/explanations.")
        return

    default_file = os.environ.get("KAIROS_EXPLANATION_JSON")
    default_index = 0
    if default_file:
        default_path = Path(default_file).resolve()
        for idx, candidate in enumerate(files):
            if candidate.resolve() == default_path:
                default_index = idx
                break

    selection = st.sidebar.selectbox("Explanation JSON", files, index=default_index, format_func=lambda p: p.name)

    try:
        payload = load_payload(selection)
    except Exception as exc:  # pragma: no cover
        st.error(f"Failed to load payload: {exc}")
        return

    graph_rows = summarise_graph_edges(payload, node_map)
    node_entries = payload.get("nodes", []) or []

    window_name = payload.get("window_path", selection.name)
    events = payload.get("num_events", 0)
    threshold = float(payload.get("threshold", 0.0))
    gpt_summary = payload.get("gpt_summary")

    node_rows = summarise_nodes(payload, node_map, float(payload.get("threshold", 0.0)))
    st.sidebar.metric("Nodes", len(node_rows))
    st.sidebar.metric("Graph edges", len(graph_rows))
    st.sidebar.metric("GPT summary", "Yes" if gpt_summary else "No")

    st.markdown(f"### Window: `{window_name}`")
    col1, col2, col3 = st.columns(3)
    col1.metric("Events", f"{events:,}")
    col2.metric("Graph edges", len(graph_rows))
    col3.metric("Threshold", f"{threshold:.4f}")

    # Only Overview, with 3 visuals
    gpt_sections = parse_gpt_sections(gpt_summary)

    st.markdown("#### Top GraphMask edges")
    cols = st.columns([2, 1])
    with cols[0]:
        if graph_rows:
            st.plotly_chart(plot_top_edges_bar(graph_rows, k=10), use_container_width=True)
        else:
            st.info("No GraphMask edges available.")
    with cols[1]:
        text = gpt_sections.get("Why flagged?") or gpt_sections.get("What happened?")
        if text:
            st.markdown("**Why flagged?**")
            st.markdown(text)

    st.markdown("#### Top nodes by average loss")
    if node_rows:
        st.plotly_chart(plot_node_scores_bar(node_rows, threshold, k=10), use_container_width=True)
    else:
        st.info("No nodes present.")

    text = gpt_sections.get("Who's involved?")
    if text:
        st.markdown("#### Who's involved?")
        st.markdown(text)

    st.markdown("#### Event volume over time (aggregate timestamps)")
    cols = st.columns([2, 1])
    with cols[0]:
        timeline_fig = plot_event_timeline(payload)
        if timeline_fig.data:
            st.plotly_chart(timeline_fig, use_container_width=True)
        else:
            st.info("No timestamps available for timeline.")
    with cols[1]:
        text = gpt_sections.get("What happened?")
        if text:
            st.markdown("**What happened?**")
            st.markdown(text)

    with st.expander("Raw explanation payload"):
        st.json(payload)


if __name__ == "__main__":
    main()
