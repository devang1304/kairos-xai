"""
Explanation pipeline focused on attack windows for the Kairos TGN model.

Workflow (no CLI arguments):
  * Load the pretrained model.
  * Locate attack windows using ``fetch_attack_list``.
  * For each window:
      - Aggregate high-loss events to produce a GraphMask-style graph story.
      - Select nodes whose cumulative loss exceeds the threshold.
      - Run GNNExplainer and VA-TGExplainer on edges touching those nodes.
  * Persist concise JSON outputs under ``artifact/explanations``.
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

import torch
from tqdm import tqdm

try:
    from ..config import ARTIFACT_DIR, include_edge_type, node_embedding_dim
except ImportError:  # pragma: no cover
    from config import ARTIFACT_DIR, include_edge_type, node_embedding_dim

from . import gnn_explainer, graphmask_explainer, utils, va_tg_explainer
from .utils import TemporalLinkWrapper, ensure_gpu_space, log_cuda_memory

DEFAULT_GRAPH_LABEL = "4_6"
USE_ATTACK_WINDOWS = True  # Set False to process the entire day (edit this flag manually).
MAX_EVENTS_PER_WINDOW = 50
GRAPHMASK_TOP_EVENTS = 25
MAX_NODES_PER_WINDOW = 20
THRESHOLD_MULTIPLIER = 1.5
TOP_K_EDGE_EXPLANATIONS = 10
MIN_EDGE_WEIGHT = 0.1
MIN_NODE_SCORE = 0.1

OUTPUT_DIR = os.path.join(ARTIFACT_DIR, "explanations")


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("explanations_logger")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    handler = logging.FileHandler(os.path.join(OUTPUT_DIR, "temporal_explanations.log"))
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


def _select_nodes(contexts: List[utils.EventContext], threshold: float) -> Tuple[List[int], Dict[int, float]]:
    scores = utils.aggregate_node_scores(contexts)
    selected = [node for node, score in scores.items() if score >= threshold]
    if selected:
        return selected[:MAX_NODES_PER_WINDOW], scores

    fallback = [
        node for node, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if score >= MIN_NODE_SCORE
    ]
    return fallback[:MAX_NODES_PER_WINDOW], scores


def _serialise_tensor(tensor: torch.Tensor | None) -> List[float]:
    if tensor is None:
        return []
    return tensor.detach().cpu().tolist()


def _serialise_edge_map(edge_map: Dict[Tuple[int, int, str], Dict[str, float | int | List[int]]]) -> List[Dict[str, object]]:
    return [
        {
            "src": key[0],
            "dst": key[1],
            "relation": key[2],
            "weight": value.get("weight", 0.0),
            "count": value.get("count", 0),
            "timestamps": value.get("timestamps", []),
        }
        for key, value in edge_map.items()
    ]


def _top_edge_explanations(
    mask_metrics: Dict[str, float],
    context: utils.EventContext,
) -> List[Dict[str, float]]:
    weights = mask_metrics.get("edge_mask", [])
    if not weights:
        return []
    ranked = sorted(
        zip(range(len(weights)), weights),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top = []
    for idx, weight in ranked:
        if len(top) >= TOP_K_EDGE_EXPLANATIONS:
            break
        if weight < MIN_EDGE_WEIGHT:
            continue
        src = int(context.edge_index[0, idx])
        dst = int(context.edge_index[1, idx])
        timestamp = int(context.edge_times[idx].item())
        msg_slice = context.edge_messages[idx]
        relation_idx = torch.argmax(msg_slice[node_embedding_dim:-node_embedding_dim]).item()
        relation = include_edge_type[relation_idx]
        top.append(
            {
                "src": src,
                "dst": dst,
                "relation": relation,
                "timestamp": timestamp,
                "weight": float(weight),
            }
        )
    return top


def _collect_windows(train_data, memory, gnn, link_pred, device):
    windows: List[Tuple[int, int, str]]
    if USE_ATTACK_WINDOWS:
        windows = [
            interval
            for interval in utils.load_attack_intervals()
            if f"graph_{DEFAULT_GRAPH_LABEL}" in interval[2]
        ]
    else:
        start_ns = int(train_data.t.min().item())
        end_ns = int(train_data.t.max().item())
        windows = [(start_ns, end_ns, f"graph_{DEFAULT_GRAPH_LABEL}_full_day")]
    return utils.group_contexts_by_windows(train_data, memory, gnn, link_pred, windows, device=device)


def run_pipeline() -> Dict[str, object]:
    logger = _setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")
    storage_mode = utils._CONTEXT_STORAGE_MODE  # type: ignore[attr-defined]
    if storage_mode == "auto":
        storage_mode = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"[info] Context tensors will be cached on {storage_mode.upper()} (override via KAIROS_CONTEXT_DEVICE).")
    print(f"[info] Loading Kairos model and graph for label '{DEFAULT_GRAPH_LABEL}'...")
    memory, gnn, link_pred = utils.load_model(device=device)
    train_data = utils.load_temporal_graph(DEFAULT_GRAPH_LABEL)
    print("[success] Model and graph loaded.")

    print("[info] Gathering attack windows and streaming contexts...")
    window_contexts = _collect_windows(train_data, memory, gnn, link_pred, device)
    num_windows = len(window_contexts)
    total_events = sum(len(ctxs) for ctxs in window_contexts.values())
    print(f"[info] Collected contexts for {num_windows} window(s); total events: {total_events}.")

    masker = graphmask_explainer.GraphMaskExplainer()
    temporal_explainer = va_tg_explainer.VATGExplainer()
    print("[info] Initialised GraphMask and VA-TG explainers.")

    outputs: List[Dict[str, object]] = []

    for window, contexts in tqdm(window_contexts.items(), desc="Windows", leave=True):
        if not contexts:
            print(f"[warn] Window {window[2]} has no contexts; skipping.")
            continue

        print(f"[info] Processing window {window[2]} with {len(contexts)} events...")
        contexts = sorted(contexts, key=lambda ctx: ctx.loss, reverse=True)
        losses = [ctx.loss for ctx in contexts]
        threshold = _compute_threshold(losses)

        high_loss_events = [(idx, ctx) for idx, ctx in enumerate(contexts) if ctx.loss >= threshold]
        high_loss_events = high_loss_events[:MAX_EVENTS_PER_WINDOW]

        if not high_loss_events:
            high_loss_events = list(enumerate(contexts[:MAX_EVENTS_PER_WINDOW]))

        def _wrapper_factory(ctx: utils.EventContext) -> TemporalLinkWrapper:
            return TemporalLinkWrapper(gnn, link_pred, ctx, device)

        mask_results = masker.explain_window(
            high_loss_events,
            wrapper_factory=_wrapper_factory,
            device=device,
            top_k_events=GRAPHMASK_TOP_EVENTS,
        )
        print(f"[info] GraphMask analysed {len(mask_results)} event(s) for window {window[2]}.")
        aggregated_map = graphmask_explainer.GraphMaskExplainer.aggregate(mask_results)

        selected_nodes, node_scores = _select_nodes(contexts, threshold)
        print(f"[info] Selected {len(selected_nodes)} node(s) for detailed explanations.")

        node_outputs = []
        for node in tqdm(selected_nodes, desc="Nodes", leave=False):
            related_contexts = [ctx for ctx in contexts if ctx.src_node == node or ctx.dst_node == node]
            related_contexts = related_contexts[:MAX_EVENTS_PER_WINDOW]

            gnn_results = []
            va_event_results: List[va_tg_explainer.VATGResult] = []
            va_serialised = []

            pending: Deque[utils.EventContext] = deque(related_contexts)
            deferred_attempted = False  # ensure we only retry once after clearing cache

            while pending:
                ctx = pending.popleft()
                if not ensure_gpu_space():
                    if not deferred_attempted:
                        print("[warn] Low GPU memory; retrying node contexts after clearing cache.")
                        torch.cuda.empty_cache()
                        pending.appendleft(ctx)
                        deferred_attempted = True
                        continue
                    print("[warn] Persistently low GPU memory; proceeding with GNNExplainer anyway.")

                log_cuda_memory(f"GNNExplainer event {ctx.event_index}")
                gnn_metrics = gnn_explainer.explain_event(ctx, gnn, link_pred, device)
                gnn_results.append(
                    {
                        "event_index": ctx.event_index,
                        "src_node": ctx.src_node,
                        "dst_node": ctx.dst_node,
                        "prob_full": gnn_metrics["prob_full"],
                        "prob_keep": gnn_metrics["prob_keep"],
                        "prob_drop": gnn_metrics["prob_drop"],
                        "comprehensiveness": gnn_metrics["comprehensiveness"],
                        "sufficiency": gnn_metrics["sufficiency"],
                        "sparsity": gnn_metrics["sparsity"],
                        "entropy": gnn_metrics["entropy"],
                        "runtime_sec": gnn_metrics["runtime_sec"],
                        "kept_edges": gnn_metrics["kept_edges"],
                        "top_edges": _top_edge_explanations(gnn_metrics, ctx),
                    }
                )

                wrapper = TemporalLinkWrapper(gnn, link_pred, ctx, device)
                log_cuda_memory(f"VA-TG event {ctx.event_index}")
                if not ensure_gpu_space():
                    if not deferred_attempted:
                        print("[warn] Low GPU memory before VA-TG; retrying after cache clear.")
                        torch.cuda.empty_cache()
                        if ensure_gpu_space():
                            deferred_attempted = True
                        else:
                            print("[warn] Still low after retry; running VA-TG explainer anyway.")
                            deferred_attempted = True
                    else:
                        print("[warn] Persistently low GPU memory; continuing with VA-TG explainer.")

                va_result = temporal_explainer.explain_event(ctx, wrapper, device)
                va_event_results.append(va_result)
                va_serialised.append(
                    {
                        "event_index": va_result.event_index,
                        "edge_importance": _serialise_tensor(va_result.edge_importance),
                        "edges": [
                            {"src": edge[0], "dst": edge[1], "relation": edge[2], "timestamp": edge[3]}
                            for edge in va_result.edges
                        ],
                        "kl_history": va_result.kl_history,
                        "loss_history": va_result.loss_history,
                    }
                )

                if not pending and deferred:
                    print("[warn] Processing deferred node contexts after freeing GPU memory.")
                    torch.cuda.empty_cache()
                    pending = deferred
                    deferred = deque()

            va_aggregate = va_tg_explainer.VATGExplainer.aggregate(va_event_results)

            node_outputs.append(
                {
                    "node_id": node,
                    "score": node_scores.get(node, 0.0),
                    "gnn": gnn_results,
                    "va_tg": {
                        "events": va_serialised,
                        "aggregate": _serialise_edge_map(va_aggregate),
                    },
                }
            )

        window_output = {
            "window_path": window[2],
            "start_ns": window[0],
            "end_ns": window[1],
            "threshold": threshold,
            "num_events": len(contexts),
            "graphmask": {
                "per_event": [
                    {
                        "event_index": res.event_index,
                        "edge_importance": _serialise_tensor(res.edge_importance),
                        "edges": [
                            {"src": edge[0], "dst": edge[1], "relation": edge[2], "timestamp": edge[3]}
                            for edge in res.edges
                        ],
                        "loss_history": res.loss_history,
                    }
                    for res in mask_results
                ],
                "aggregate": _serialise_edge_map(aggregated_map),
            },
            "nodes": node_outputs,
        }

        outputs.append(window_output)
        logger.info("Window %s | events=%d | threshold=%.4f", window[2], len(contexts), threshold)
        print(f"[success] Completed explanations for window {window[2]} (threshold={threshold:.4f}).")

        out_path = os.path.join(
            OUTPUT_DIR,
            f"{os.path.basename(window[2]).replace('.txt', '')}_explanations.json",
        )
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(window_output, fh, indent=2)
        print(f"[success] Wrote window explanations to {out_path}")

    summary = {"graph_label": DEFAULT_GRAPH_LABEL, "windows": outputs}
    summary_path = os.path.join(OUTPUT_DIR, f"graph_{DEFAULT_GRAPH_LABEL}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[success] Summary saved to {summary_path}")

    return summary


def main() -> None:
    summary = run_pipeline()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
