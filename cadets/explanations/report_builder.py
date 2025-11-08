"""Generate Markdown/HTML analyst reports from explanation artifacts."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytz

try:  # Optional GPT dependency
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

EST = pytz.timezone("US/Eastern")


@dataclass
class EdgeSummary:
    src: str
    dst: str
    relation: str
    weight: float
    count: int
    timestamps: List[str]


def _to_local_time(ns: int) -> str:
    seconds, nanos = divmod(int(ns), 1_000_000_000)
    dt = pytz.UTC.localize(datetime.utcfromtimestamp(seconds)).astimezone(EST)
    return f"{dt.strftime('%Y-%m-%d %H:%M:%S')}.{str(nanos).zfill(9)}"


def _resolve_node(node_id: int, node_map: Dict[int, str]) -> str:
    label = node_map.get(node_id)
    return label if label else f"Node {node_id}"


def _summarise_edges(aggregate: List[Dict[str, object]], node_map: Dict[int, str]) -> List[EdgeSummary]:
    items: List[EdgeSummary] = []
    for entry in aggregate:
        src = _resolve_node(entry["src"], node_map)
        dst = _resolve_node(entry["dst"], node_map)
        relation = entry.get("relation", "?")
        weight = float(entry.get("weight", 0.0))
        count = int(entry.get("count", 0))
        timestamps = [_to_local_time(ts) for ts in entry.get("timestamps", [])[:5]]
        items.append(EdgeSummary(src, dst, relation, weight, count, timestamps))
    items.sort(key=lambda e: e.weight, reverse=True)
    return items


def _load_node_map(mapping_path: Optional[Path]) -> Dict[int, str]:
    if mapping_path and mapping_path.exists():
        data = json.loads(mapping_path.read_text(encoding="utf-8"))
        return {int(k): v for k, v in data.items()}
    return {}


def _prepare_node_entries(nodes: List[Dict[str, object]], node_map: Dict[int, str]) -> List[Dict[str, object]]:
    prepared: List[Dict[str, object]] = []
    for node in nodes:
        label = _resolve_node(node["node_id"], node_map)
        gnn_edges: List[Dict[str, object]] = []
        for gnn_event in node.get("gnn", []):
            for edge in gnn_event.get("top_edges", []):
                gnn_edges.append(
                    {
                        "src": _resolve_node(edge["src"], node_map),
                        "dst": _resolve_node(edge["dst"], node_map),
                        "relation": edge.get("relation", "?"),
                        "weight": edge.get("weight", 0.0),
                    }
                )
        gnn_edges.sort(key=lambda e: e["weight"], reverse=True)

        va = node.get("va_tg", {})
        va_edges = [
            {
                "src": _resolve_node(entry["src"], node_map),
                "dst": _resolve_node(entry["dst"], node_map),
                "relation": entry.get("relation", "?"),
                "weight": entry.get("weight", 0.0),
            }
            for entry in va.get("aggregate", [])
        ]
        va_edges.sort(key=lambda e: e["weight"], reverse=True)

        prepared.append(
            {
                "label": label,
                "score": node.get("score", 0.0),
                "gnn_edges": gnn_edges,
                "va_edges": va_edges,
            }
        )
    prepared.sort(key=lambda e: e["score"], reverse=True)
    return prepared


def _call_gpt(summary_payload: Dict[str, object]) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    prompt = (
        "You are an incident response analyst. Produce a detailed, digestible report that summarizes the key attack patterns described in the explanations JSON. Keep every finding grounded strictly in the evidence provided by that JSON."
    )
    result = client.responses.create(
        model="gpt-5",
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(summary_payload)},
        ],
        max_output_tokens=400,
    )
    try:
        return result.output_text  # type: ignore[attr-defined]
    except AttributeError:  # pragma: no cover
        return None


def _render_markdown(context: Dict[str, object]) -> str:
    lines = [
        "# Kairos Explanation Report",
        "",
        f"**Window**: {context['window_path']}  ",
        f"**Events**: {context['num_events']}  ",
        f"**Threshold**: {context['threshold']:.4f}",
        "",
        "## Graph-Level Highlights (GraphMask)",
    ]
    for edge in context["graph_edges"][:10]:
        lines.append(
            f"- **{edge.relation}**: {edge.src} → {edge.dst} (weight={edge.weight:.3f}, count={edge.count})"
        )
        if edge.timestamps:
            lines.append(f"  - Times: {', '.join(edge.timestamps)}")
    lines.append("")
    lines.append("## Top Nodes")
    for node in context["nodes"][:10]:
        lines.append(f"### {node['label']} (score={node['score']:.2f})")
        if node["gnn_edges"]:
            top = ", ".join(
                f"{edge['relation']} {edge['src']} → {edge['dst']} ({edge['weight']:.3f})"
                for edge in node["gnn_edges"][:3]
            )
            lines.append(f"- GNN top edges: {top}")
        if node["va_edges"]:
            top = ", ".join(
                f"{edge['relation']} {edge['src']} → {edge['dst']} ({edge['weight']:.3f})"
                for edge in node["va_edges"][:3]
            )
            lines.append(f"- VA aggregate edges: {top}")
        lines.append("")
    if context.get("gpt_summary"):
        lines.append("## GPT Narrative")
        lines.append(str(context["gpt_summary"]))
    return "\n".join(lines)


def _render_html(context: Dict[str, object]) -> str:
    head = """<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"""
    head += "<title>Kairos Explanation Report</title>"
    head += "<style>body{font-family:Arial, sans-serif;line-height:1.6;margin:2rem;}h1,h2,h3{color:#2c3e50;}" \
            ".edge,.node{margin-bottom:1rem;} .timestamps{font-size:0.9em;color:#555;}</style></head><body>"
    body = [head, "<h1>Kairos Explanation Report</h1>"]
    body.append(
        f"<p><strong>Window:</strong> {context['window_path']}<br/>"
        f"<strong>Events:</strong> {context['num_events']}<br/>"
        f"<strong>Threshold:</strong> {context['threshold']:.4f}</p>"
    )
    body.append("<h2>Graph-Level Highlights (GraphMask)</h2>")
    for edge in context["graph_edges"][:10]:
        snippet = (
            f"<div class='edge'><strong>{edge.relation}</strong>: {edge.src} → {edge.dst} "
            f"(weight={edge.weight:.3f}, count={edge.count})"
        )
        if edge.timestamps:
            snippet += f"<div class='timestamps'>Times: {', '.join(edge.timestamps)}</div>"
        snippet += "</div>"
        body.append(snippet)
    body.append("<h2>Top Nodes</h2>")
    for node in context["nodes"][:10]:
        body.append(f"<div class='node'><h3>{node['label']} (score={node['score']:.2f})</h3>")
        if node["gnn_edges"]:
            edges = ", ".join(
                f"{edge['relation']} {edge['src']} → {edge['dst']} ({edge['weight']:.3f})"
                for edge in node["gnn_edges"][:3]
            )
            body.append(f"<p><strong>GNN top edges:</strong> {edges}</p>")
        if node["va_edges"]:
            edges = ", ".join(
                f"{edge['relation']} {edge['src']} → {edge['dst']} ({edge['weight']:.3f})"
                for edge in node["va_edges"][:3]
            )
            body.append(f"<p><strong>VA aggregate edges:</strong> {edges}</p>")
        body.append("</div>")
    if context.get("gpt_summary"):
        body.append("<h2>GPT Narrative</h2>")
        body.append(f"<p>{context['gpt_summary']}</p>")
    body.append("</body></html>")
    return "".join(body)


def build_reports(
    report_json: Dict[str, object],
    output_dir: Path,
    node_mapping_path: Optional[Path] = None,
    run_gpt: bool = True,
) -> Tuple[Path, Path, Optional[str]]:
    node_map = _load_node_map(node_mapping_path)
    graph_edges = _summarise_edges(report_json["graphmask"]["aggregate"], node_map)
    node_entries = _prepare_node_entries(report_json.get("nodes", []), node_map)

    summary_payload = {
        "window": report_json["window_path"],
        "top_graph_edges": [edge.__dict__ for edge in graph_edges[:5]],
        "top_nodes": [
            {
                "label": node["label"],
                "score": node["score"],
                "top_edges": node["gnn_edges"][:3],
            }
            for node in node_entries[:5]
        ],
    }
    gpt_summary = _call_gpt(summary_payload) if run_gpt else None

    context = {
        "window_path": report_json["window_path"],
        "num_events": report_json["num_events"],
        "threshold": report_json["threshold"],
        "graph_edges": graph_edges,
        "nodes": node_entries,
        "gpt_summary": gpt_summary,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = Path(report_json["window_path"])
    base_name = raw_path.stem.replace(":", "_").replace("/", "_")
    if not base_name:
        base_name = "window"
    md_path = output_dir / f"{base_name}_report.md"
    html_path = output_dir / f"{base_name}_report.html"

    md_path.write_text(_render_markdown(context), encoding="utf-8")
    html_path.write_text(_render_html(context), encoding="utf-8")

    return md_path, html_path, gpt_summary
