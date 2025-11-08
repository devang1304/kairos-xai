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

import copy
import json
import logging
import os
from pathlib import Path
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

import torch
from tqdm import tqdm

try:
    from ..config import ARTIFACT_DIR, NODE_MAPPING_JSON, include_edge_type, node_embedding_dim
    from ..kairos_utils import datetime_to_ns_time_US, ns_time_to_datetime_US
except ImportError:  # pragma: no cover
    from config import ARTIFACT_DIR, NODE_MAPPING_JSON, include_edge_type, node_embedding_dim
    from kairos_utils import datetime_to_ns_time_US, ns_time_to_datetime_US

from . import gnn_explainer, graphmask_explainer, report_builder, utils, va_tg_explainer
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
WARMUP_MARGIN_SECONDS = int(os.environ.get("KAIROS_EXPLAIN_WARMUP_SEC", 2 * 3600))

HARD_CODED_ATTACK_WINDOWS = [
    ("2018-04-06 11:00:00", "2018-04-06 12:15:00"),
]

OUTPUT_DIR = os.path.join(ARTIFACT_DIR, "explanations")
NODE_MAPPING_PATH = os.getenv("KAIROS_NODE_MAPPING_JSON", NODE_MAPPING_JSON)


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
    if USE_ATTACK_WINDOWS:
        windows: List[Tuple[int, int, str]] = []
        for start_str, end_str in HARD_CODED_ATTACK_WINDOWS:
            start_ns = datetime_to_ns_time_US(start_str)
            end_ns = datetime_to_ns_time_US(end_str)
            identifier = f"hardcoded/{start_str.replace(' ', '_')}~{end_str.replace(' ', '_')}.txt"
            windows.append((start_ns, end_ns, identifier))
    else:
        start_ns = int(train_data.t.min().item())
        end_ns = int(train_data.t.max().item())
        windows = [(start_ns, end_ns, f"graph_{DEFAULT_GRAPH_LABEL}_full_day")]

    cached = utils.load_context_cache(DEFAULT_GRAPH_LABEL, windows)
    if cached is not None:
        print("[info] Loaded cached event contexts for selected windows.")
        return cached

    if not windows:
        return {}

    earliest_start = min(start for start, _, _ in windows)
    latest_end = max(end for _, end, _ in windows)

    margin_ns = WARMUP_MARGIN_SECONDS * 1_000_000_000
    data_min = int(train_data.t.min().item())
    data_max = int(train_data.t.max().item())
    slice_start_ns = max(data_min, earliest_start - margin_ns)
    slice_end_ns = min(data_max, latest_end)

    print(
        f"[info] Streaming slice from {ns_time_to_datetime_US(slice_start_ns)} "
        f"to {ns_time_to_datetime_US(slice_end_ns)} for context collection."
    )

    sliced_data, _, _ = utils.slice_temporal_graph(train_data, slice_start_ns, slice_end_ns)
    context_start_offset = int(
        torch.searchsorted(sliced_data.t, torch.tensor(earliest_start, device=sliced_data.t.device))
    )

    contexts_by_window: Dict[Tuple[int, int, str], List[utils.EventContext]] = {window: [] for window in windows}

    def _predicate(ctx: utils.EventContext) -> bool:
        included = False
        for window in windows:
            start_ns, end_ns, _ = window
            if start_ns <= ctx.timestamp <= end_ns:
                contexts_by_window[window].append(ctx)
                included = True
        return included

    # Exhaust generator to populate contexts_by_window; actual yielded values aren't needed.
    for _ in utils.stream_event_contexts(
        sliced_data,
        memory,
        gnn,
        link_pred,
        device=device,
        start_offset=context_start_offset,
        predicate=_predicate,
    ):
        pass

    print("[info] Persisting window contexts to cache for future runs.")
    utils.save_context_cache(DEFAULT_GRAPH_LABEL, windows, contexts_by_window)
    return contexts_by_window


def run_pipeline() -> Dict[str, object]:
    logger = _setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")
    configured_mode = utils._CONTEXT_STORAGE_MODE  # type: ignore[attr-defined]
    effective_mode = "cpu" if configured_mode == "auto" else configured_mode
    print(
        f"[info] Context tensors will be cached on {effective_mode.upper()} "
        "(override via KAIROS_CONTEXT_DEVICE)."
    )
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
    cpu_device = torch.device("cpu")
    gnn_cpu = copy.deepcopy(gnn).to(cpu_device)
    link_pred_cpu = copy.deepcopy(link_pred).to(cpu_device)
    temporal_explainer_cpu = va_tg_explainer.VATGExplainer()
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

        def _wrapper_factory(ctx: utils.EventContext, target_device: torch.device) -> TemporalLinkWrapper:
            if target_device.type == "cpu":
                return TemporalLinkWrapper(gnn_cpu, link_pred_cpu, ctx, target_device)
            return TemporalLinkWrapper(gnn, link_pred, ctx, target_device)

        mask_results = masker.explain_window(
            high_loss_events,
            wrapper_factory=_wrapper_factory,
            device=device,
            top_k_events=GRAPHMASK_TOP_EVENTS,
            fallback_wrapper_factory=_wrapper_factory,
            fallback_device=cpu_device,
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
            gnn_event_counter = 0

            while pending:
                ctx = pending.popleft()
                target_device = device
                gnn_module = gnn
                link_module = link_pred
                explainer_instance = temporal_explainer

                if not ensure_gpu_space():
                    print(f"[warn] Low GPU memory for event {ctx.event_index}; falling back to CPU.")
                    target_device = cpu_device
                    gnn_module = gnn_cpu
                    link_module = link_pred_cpu
                    explainer_instance = temporal_explainer_cpu

                gnn_event_counter += 1
                log_cuda_memory(f"GNNExplainer event {ctx.event_index}", step=gnn_event_counter)
                gnn_metrics = gnn_explainer.explain_event(ctx, gnn_module, link_module, target_device)
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

                wrapper = TemporalLinkWrapper(gnn_module, link_module, ctx, target_device)
                log_cuda_memory(f"VA-TG event {ctx.event_index}", step=gnn_event_counter)
                if target_device.type != "cpu" and not ensure_gpu_space():
                    print(f"[warn] Low GPU memory before VA-TG for event {ctx.event_index}; retrying on CPU.")
                    wrapper = TemporalLinkWrapper(gnn_cpu, link_pred_cpu, ctx, cpu_device)
                    explainer_instance = temporal_explainer_cpu
                    target_device = cpu_device

                va_result = explainer_instance.explain_event(ctx, wrapper, target_device)
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

        try:
            mapping_path = Path(NODE_MAPPING_PATH) if NODE_MAPPING_PATH else None
            md_path, html_path, _ = report_builder.build_reports(
                window_output,
                Path(OUTPUT_DIR),
                node_mapping_path=mapping_path,
                run_gpt=True,
            )
            logger.info("Analyst reports generated: %s, %s", md_path.name, html_path.name)
        except Exception as report_err:
            logger.warning("Report generation failed for %s: %s", window[2], report_err)
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
