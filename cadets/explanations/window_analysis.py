"""
Simplified explanation pipeline for the Kairos TGN model.

Run:
    python -m cadets.explanations.window_analysis

The script will:
  * Load the trained Kairos model.
  * Train PGExplainer on a sample of April 6th events.
  * Use `fetch_attack_list()` to locate the known attack windows.
  * For each attack window, run PGExplainer on every event and GNNExplainer
    on the high-loss subset (Kairos-style threshold).
  * Log window-level summaries to `artifact/explanations.log` and print an
    aggregated JSON report to stdout.
"""

import json
import logging
import os
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

try:
    from ..config import ARTIFACT_DIR
except ImportError:  # pragma: no cover
    from config import ARTIFACT_DIR

from . import gnn_explainer, pg_explainer, utils

DEFAULT_GRAPH_LABEL = "4_6"
PG_TRAIN_SAMPLE = 1000
MAX_GNN_EVENTS = 50
THRESHOLD_MULTIPLIER = 1.5


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("explanations_logger")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    handler = logging.FileHandler(os.path.join(ARTIFACT_DIR, "explanations.log"))
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def _compute_threshold(losses: List[float]) -> float:
    tensor = torch.tensor(losses)
    mu = float(tensor.mean().item())
    sigma = float(tensor.std(unbiased=False).item())
    return mu + THRESHOLD_MULTIPLIER * sigma


def _aggregate(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    summary: Dict[str, float] = {}
    for key in keys:
        values = [m[key] for m in metrics if key in m]
        if values:
            summary[key] = float(sum(values) / len(values))
    return summary


def run_pipeline() -> Dict[str, object]:
    logger = _setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    memory, gnn, link_pred = utils.load_model(device=device)

    # Train PGExplainer on a sample of events.
    train_data = utils.load_temporal_graph(DEFAULT_GRAPH_LABEL)
    train_contexts = utils.collect_contexts(
        train_data,
        memory,
        gnn,
        link_pred,
        limit=PG_TRAIN_SAMPLE if PG_TRAIN_SAMPLE > 0 else None,
        device=device,
    )
    if not train_contexts:
        raise RuntimeError(
            "PGExplainer training aborted: no events available in "
            f"graph_{DEFAULT_GRAPH_LABEL}.TemporalData.simple. "
            "Ensure embeddings exist before running window analysis."
        )
    pg_indices = sorted(train_contexts.keys())
    pg_algo, _ = pg_explainer.train_pg_explainer(
        memory,
        gnn,
        link_pred,
        train_data,
        pg_indices,
        contexts=train_contexts,
        device=device,
    )

    # Filter attack windows relevant to this dataset.
    attack_windows = [
        interval
        for interval in utils.load_attack_intervals()
        if f"graph_{DEFAULT_GRAPH_LABEL}" in interval[2]
    ]

    def _timestamp_in_any_window(timestamp: int) -> bool:
        for start_ns, end_ns, _ in attack_windows:
            if start_ns <= timestamp <= end_ns:
                return True
        return False

    attack_contexts = utils.collect_contexts(
        train_data,
        memory,
        gnn,
        link_pred,
        predicate=lambda ctx: _timestamp_in_any_window(ctx.timestamp),
        device=device,
    )

    window_contexts_map: Dict[Tuple[int, int, str], Dict[int, utils.EventContext]] = {
        window: {} for window in attack_windows
    }
    for idx, context in tqdm(list(attack_contexts.items()), desc="Indexing attack contexts", leave=False):
        for window in attack_windows:
            start_ns, end_ns, _ = window
            if start_ns <= context.timestamp <= end_ns:
                window_contexts_map[window][idx] = context
                context.is_attack = True

    window_summaries: List[Dict[str, object]] = []

    for start_ns, end_ns, path in tqdm(attack_windows, desc="Analyzing attack windows", leave=False):
        contexts = window_contexts_map.get((start_ns, end_ns, path), {})
        if not contexts:
            continue

        losses = [ctx.loss for ctx in contexts.values()]
        threshold = _compute_threshold(losses)
        loss_tensor = torch.tensor(losses)
        loss_mean = float(loss_tensor.mean().item())
        loss_std = float(loss_tensor.std(unbiased=False).item())
        flagged = sum(1 for loss in losses if loss > threshold)

        sorted_events: List[Tuple[int, utils.EventContext]] = sorted(
            contexts.items(), key=lambda kv: kv[1].loss, reverse=True
        )

        pg_metrics: List[Dict[str, float]] = []
        gnn_metrics: List[Dict[str, float]] = []

        window_name = os.path.basename(path)

        for _, context in tqdm(sorted_events, desc=f"PG explainer {window_name}", leave=False):
            pg_metrics.append(
                pg_explainer.explain_event(context, gnn, link_pred, pg_algo, device)
            )

        high_loss_events = [
            (idx, ctx) for idx, ctx in sorted_events if ctx.loss > threshold
        ]
        if MAX_GNN_EVENTS > 0:
            high_loss_events = high_loss_events[:MAX_GNN_EVENTS]

        for _, context in tqdm(high_loss_events, desc=f"GNN explainer {window_name}", leave=False):
            gnn_metrics.append(
                gnn_explainer.explain_event(context, gnn, link_pred, device)
            )

        summary = {
            "window": os.path.basename(path),
            "start_ns": start_ns,
            "end_ns": end_ns,
            "num_events": len(sorted_events),
            "loss_mean": loss_mean,
            "loss_std": loss_std,
            "threshold": threshold,
            "kairos_flagged": flagged,
            "kairos_flag_fraction": flagged / max(len(sorted_events), 1),
            "pg_summary": _aggregate(pg_metrics),
            "gnn_summary": _aggregate(gnn_metrics),
        }
        window_summaries.append(summary)

        logger.info("Window %s", summary["window"])
        logger.info(
            "  events=%d loss_mean=%.4f loss_std=%.4f threshold=%.4f flagged=%d",
            summary["num_events"],
            summary["loss_mean"],
            summary["loss_std"],
            summary["threshold"],
            summary["kairos_flagged"],
        )
        logger.info("  PG metrics: %s", summary["pg_summary"])
        logger.info("  GNN metrics: %s", summary["gnn_summary"])

    overall_pg = _aggregate([w["pg_summary"] for w in window_summaries if w["pg_summary"]])
    overall_gnn = _aggregate([w["gnn_summary"] for w in window_summaries if w["gnn_summary"]])

    logger.info("Overall PG summary: %s", overall_pg)
    logger.info("Overall GNN summary: %s", overall_gnn)

    return {
        "graph_label": DEFAULT_GRAPH_LABEL,
        "windows": window_summaries,
        "overall_pg": overall_pg,
        "overall_gnn": overall_gnn,
    }


def main() -> None:
    summary = run_pipeline()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
