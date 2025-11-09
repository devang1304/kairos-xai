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


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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
            name="Weight",
            showlegend=True,
        )
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text="Top GraphMask edges", x=0.5, xanchor="center"),
        xaxis_title="Weight",
        yaxis_title="Edge (relation src → dst)",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )
    fig.update_yaxes(automargin=True)
    return fig


def plot_node_scores_bar(node_rows: List[Dict[str, object]], threshold: float, k: int = 10) -> go.Figure:
    top = node_rows[:k]
    labels = [r["label"] for r in top]
    scores = [float(r["avg_loss"]) for r in top]
    colors = ["#d62728" if r.get("above") else "#2ca02c" for r in top]
    fig = go.Figure(
        go.Bar(
            x=scores,
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="avg_loss=%{x:.3f}<extra></extra>",
            name="Avg loss",
            showlegend=True,
        )
    )
    # Add a legendable threshold line as a separate scatter trace
    if labels:
        fig.add_trace(
            go.Scatter(
                x=[threshold, threshold],
                y=[-0.5, len(labels) - 0.5],
                mode="lines",
                line=dict(color="#7f7f7f", width=2, dash="dash"),
                name="Event‑loss threshold",
                hoverinfo="skip",
                showlegend=True,
            )
        )
    # Annotation for threshold semantics
    if labels:
        fig.add_annotation(
            x=threshold,
            y=1.02,
            xref="x",
            yref="paper",
            text="event‑loss threshold",
            showarrow=False,
            font=dict(color="#7f7f7f", size=12),
        )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text="Top nodes by average loss", x=0.5, xanchor="center"),
        xaxis_title="Average loss",
        yaxis_title="Node",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
    )
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
            # Accept int, float, and digit-like strings; skip anything else
            if isinstance(ts, str):
                if ts.isdigit():
                    ts = int(ts)
                else:
                    continue
            seconds, _ = divmod(int(ts), 1_000_000_000)
            dt = pytz.UTC.localize(datetime.utcfromtimestamp(seconds)).astimezone(EST)
            key = dt.strftime("%Y-%m-%d %H:%M")
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return go.Figure()
    items = sorted(counts.items())
    x = [k for k, _ in items]
    y = [v for _, v in items]
    fig = go.Figure(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(color="#1f77b4"),
            name="Events/min",
            showlegend=True,
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Time (minute)",
        yaxis_title="Events",
        title=dict(text="Event volume (GraphMask timestamps)", x=0.5, xanchor="center"),
    )
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


def _sanitize_base(stem: str) -> str:
    return stem.replace(":", "_").replace("/", "_") or "window"


def compute_report_md_path(payload: Dict[str, object], base_dir: Path) -> Path:
    raw = str(payload.get("window_path") or "")
    stem = _sanitize_base(Path(raw).stem)
    return (base_dir / f"{stem}_report.md").resolve()


def extract_node_md_sections(md_text: str) -> List[Tuple[str, str]]:
    """Extract (title, content) tuples for node-level subsections from report Markdown."""
    lines = md_text.splitlines()
    in_block = False
    current_title: Optional[str] = None
    buf: List[str] = []
    out: List[Tuple[str, str]] = []
    for line in lines:
        if line.startswith("## ") and "Node-Level Explanations" in line:
            in_block = True
            current_title = None
            buf = []
            continue
        if not in_block:
            continue
        if line.startswith("## ") and "Node-Level Explanations" not in line:
            # End of node section
            if current_title is not None:
                out.append((current_title, "\n".join(buf).strip()))
            break
        if line.startswith("### "):
            if current_title is not None:
                out.append((current_title, "\n".join(buf).strip()))
                buf = []
            current_title = line[4:].strip()
        else:
            if current_title is not None:
                buf.append(line)
    if in_block and current_title is not None:
        out.append((current_title, "\n".join(buf).strip()))
    return out


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
    page = st.sidebar.radio("Page", ("Window", "Node explanations"), index=0)

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

    if page == "Window":
        st.markdown(f"### Window: `{window_name}`")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Events in window", f"{events:,}")
        col2.metric("Sub-graph Nodes", len(node_rows))
        col3.metric("Sub-graph edges", len(graph_rows))
        col4.metric("Threshold", f"{threshold:.4f}")
        col5.metric("AI summary", "Yes" if gpt_summary else "No")

        st.divider()

        # Overview visuals + GPT
        gpt_sections = parse_gpt_sections(gpt_summary)

        # 1) Activity overview: narrative (left) + event volume (right)
        st.markdown("### Window activity overview")
        cols = st.columns([1, 2])
        with cols[0]:
            text = gpt_sections.get("What happened?")
            if text:
                st.markdown(text)
            else:
                st.info("No GPT narrative available for this window.")
        with cols[1]:
            timeline_fig = plot_event_timeline(payload)
            if timeline_fig.data:
                st.plotly_chart(timeline_fig, use_container_width=True)
            else:
                st.info("No timestamps available for timeline.")

        st.divider()

        # 2) Top GraphMask edges
        st.markdown("### Top GraphMask edges")
        if graph_rows:
            st.plotly_chart(plot_top_edges_bar(graph_rows, k=10), use_container_width=True)
        else:
            st.info("No GraphMask edges available.")

        st.divider()

        # 3) Top nodes by average loss
        st.markdown("### Top nodes by average loss")
        if node_rows:
            st.plotly_chart(plot_node_scores_bar(node_rows, threshold, k=10), use_container_width=True)
        else:
            st.info("No nodes present.")

        st.divider()

        # 4) Who's involved? (GPT)
        text = gpt_sections.get("Who's involved?")
        if text:
            st.markdown("#### Who's involved?")
            st.markdown(text)

        st.divider()

        # 5) Why flagged? (GPT) — emphasize readability with higher-level heading
        text = gpt_sections.get("Why flagged?")
        if text:
            st.markdown("### Why flagged?")
            st.markdown(text)

        # 6) What's missing or risky? (GPT)
        text = gpt_sections.get("What's missing or risky?")
        if text:
            st.divider()
            st.markdown("#### What's missing or risky?")
            st.markdown(text)

        # 7) What next? (GPT)
        text = gpt_sections.get("What next?")
        if text:
            st.divider()
            st.markdown("#### What next?")
            st.markdown(text)
    else:
        # Node explanations page (from report Markdown)
        report_md = compute_report_md_path(payload, EXPLANATION_DIR)
        if not report_md.exists():
            st.info(
                f"No report Markdown found (expected {report_md.name}). Generate it via: python -m reporting.generate_report"
            )
        else:
            md_text = report_md.read_text(encoding="utf-8")
            sections = extract_node_md_sections(md_text)
            if not sections:
                st.info("No node-level explanations found in the report.")
            else:
                st.markdown("### Node explanations")
                # Build a mapping from label → markdown content using the report titles
                md_index = {}
                for title, content in sections:
                    base = title.split(" (", 1)[0].strip() or title
                    md_index[base] = content

                # Build human-readable options from payload nodes using node mapping
                raw_nodes = payload.get("nodes", []) or []
                labels = []
                for n in raw_nodes:
                    nid = n.get("node_id")
                    label = resolve_label(nid, node_map)
                    labels.append((label, int(nid) if nid is not None else -1))
                # Disambiguate duplicate labels by appending node id
                label_counts = {}
                for label, _ in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
                options = []
                for label, nid in labels:
                    if label_counts.get(label, 0) > 1 and nid >= 0:
                        options.append(f"{label} [id={nid}]")
                    else:
                        options.append(label)
                # Fallback: if payload had no nodes, use titles from MD
                if not options:
                    options = [t.split(" (", 1)[0].strip() or t for t, _ in sections]

                selected = st.selectbox(
                    "Select a node",
                    sorted(set(options)),
                    index=0,
                    key="node_expl_select",
                )
                # Map selection back to base label (strip appended id, if any)
                base = selected.split(" [id=", 1)[0]
                st.markdown(f"#### {base}")
                st.markdown(md_index.get(base, "No explanation available in the report for this node."))


if __name__ == "__main__":
    main()
