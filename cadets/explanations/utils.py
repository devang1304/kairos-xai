import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, List, Tuple

import torch
from torch import Tensor
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader

try:
    from ..config import GRAPHS_DIR, MODELS_DIR, neighbor_size, node_embedding_dim
    from ..kairos_utils import tensor_find, datetime_to_ns_time_US, fetch_attack_list
    from .. import model
except ImportError:  # pragma: no cover - fallback when run as script
    from config import GRAPHS_DIR, MODELS_DIR, neighbor_size, node_embedding_dim
    from kairos_utils import tensor_find, datetime_to_ns_time_US, fetch_attack_list
    import model

# Controls where event context tensors are stored. Use "gpu", "cpu", or "cpu_pin";
# default "auto" keeps tensors on GPU when available.
_CONTEXT_STORAGE_MODE = os.environ.get("KAIROS_CONTEXT_DEVICE", "auto").lower()


def _store_tensor(tensor: Optional[Tensor]) -> Optional[Tensor]:
    if tensor is None or not torch.is_tensor(tensor):
        return tensor
    result = tensor.detach()
    target_mode = _CONTEXT_STORAGE_MODE
    if target_mode == "auto":
        target_mode = "gpu" if torch.cuda.is_available() else "cpu"
    if target_mode == "cpu" or target_mode == "cpu_pin":
        result = result.to("cpu")
        if target_mode == "cpu_pin":
            try:
                result = result.pin_memory()
            except RuntimeError:
                pass
    else:
        # Keep tensor on its current device (GPU/MPS/CPU)
        pass
    return result

@dataclass
class EventContext:
    """Snapshot of the TGN state around a single temporal event."""

    event_index: int
    src_node: int
    dst_node: int
    timestamp: int
    label: int
    memory_inputs: Tensor
    last_update: Tensor
    edge_index: Tensor
    edge_messages: Tensor
    edge_times: Tensor
    base_embeddings: Tensor
    logits: Tensor
    probabilities: Tensor
    prob_label: float
    loss: float
    src_local_index: int
    dst_local_index: int
    node_ids: Tensor
    raw_message: Tensor
    build_time: float = 0.0
    is_attack: bool = False

    def copy(self) -> "EventContext":
        return EventContext(
            event_index=self.event_index,
            src_node=self.src_node,
            dst_node=self.dst_node,
            timestamp=self.timestamp,
            label=self.label,
            memory_inputs=self.memory_inputs.clone(),
            last_update=self.last_update.clone(),
            edge_index=self.edge_index.clone(),
            edge_messages=self.edge_messages.clone(),
            edge_times=self.edge_times.clone(),
            base_embeddings=self.base_embeddings.clone(),
            logits=self.logits.clone(),
            probabilities=self.probabilities.clone(),
            prob_label=self.prob_label,
            loss=self.loss,
            src_local_index=self.src_local_index,
            dst_local_index=self.dst_local_index,
            node_ids=self.node_ids.clone(),
            raw_message=self.raw_message.clone(),
            is_attack=self.is_attack,
        )


class TemporalLinkWrapper(torch.nn.Module):
    """Wraps the Kairos GNN for compatibility with PyG explainer APIs."""

    def __init__(
        self,
        gnn: torch.nn.Module,
        link_pred: torch.nn.Module,
        context: EventContext,
        device: torch.device,
    ):
        super().__init__()
        self.gnn = gnn
        self.link_pred = link_pred
        self.src_idx = int(context.src_local_index)
        self.dst_idx = int(context.dst_local_index)

        self.register_buffer("last_update", context.last_update.detach().to(device))
        self.register_buffer("edge_times", context.edge_times.detach().to(device))
        self.register_buffer("edge_messages", context.edge_messages.detach().to(device))
        self.register_buffer("baseline_embeddings", context.base_embeddings.detach().to(device))
        self.register_buffer("node_ids", context.node_ids.detach().to(device))

        self.gnn.eval()
        self.link_pred.eval()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        edge_t: Optional[Tensor] = None,
    ) -> Tensor:
        if edge_index.numel() == 0:
            z = self.baseline_embeddings
        else:
            msg = edge_attr if edge_attr is not None else self.edge_messages
            times = edge_t if edge_t is not None else self.edge_times
            z = self.gnn(x, self.last_update, edge_index, times, msg)
        return self.link_pred(z[[self.src_idx]], z[[self.dst_idx]])


def load_temporal_graph(label: str, root: Optional[str] = None) -> TemporalData:
    """Loads a TemporalData object for the requested day/window label."""
    root = root or GRAPHS_DIR
    path = os.path.join(root, f"graph_{label}.TemporalData.simple")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Temporal graph not found at {path}")
    data: TemporalData = torch.load(path)
    return data


def load_model(
    model_path: Optional[str] = None,
    device: Optional[torch.device] = None,
):
    """Loads the trained Kairos model components."""
    model_path = model_path or os.path.join(MODELS_DIR, "models.pt")
    device = device or model.device
    memory, gnn, link_pred, _ = torch.load(model_path, map_location=device)
    memory = memory.to(device)
    gnn = gnn.to(device)
    link_pred = link_pred.to(device)

    memory.eval()
    gnn.eval()
    link_pred.eval()

    if hasattr(memory, "num_nodes"):
        model.configure_node_capacity(int(memory.num_nodes))

    return memory, gnn, link_pred


def _init_neighbor_loader(device: torch.device) -> LastNeighborLoader:
    return LastNeighborLoader(
        model.max_node_num,
        size=neighbor_size,
        device=device,
    )


def build_event_context(
    data: TemporalData,
    memory: torch.nn.Module,
    gnn: torch.nn.Module,
    link_pred: torch.nn.Module,
    event_index: int,
    device: Optional[torch.device] = None,
) -> EventContext:
    """Replays the temporal stream until `event_index` and captures state."""
    device = device or model.device
    memory.reset_state()
    loader = _init_neighbor_loader(device)
    loader.reset_state()

    temporal_loader = TemporalDataLoader(data, batch_size=1, shuffle=False)
    processed = 0

    with torch.no_grad():
        for batch in temporal_loader:
            src_cpu = batch.src
            dst_cpu = batch.dst
            t_cpu = batch.t
            msg_cpu = batch.msg

            src = src_cpu.to(device)
            dst = dst_cpu.to(device)
            t = t_cpu.to(device)
            msg = msg_cpu.to(device)

            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = loader(n_id)
            model.assoc[n_id] = torch.arange(n_id.size(0), device=device)

            memory_inputs, last_update = memory(n_id)
            e_id_cpu = e_id.cpu()
            edge_times = data.t[e_id_cpu].to(device)
            edge_messages = data.msg[e_id_cpu].to(device)

            embeddings = gnn(memory_inputs, last_update, edge_index, edge_times, edge_messages)
            logits = link_pred(embeddings[model.assoc[src]], embeddings[model.assoc[dst]])
            probabilities = logits.softmax(dim=-1)

            if processed == event_index:
                label = tensor_find(
                    msg_cpu[0][node_embedding_dim:-node_embedding_dim], 1
                ) - 1
                prob_label = probabilities[0, label].item()
                loss = -float(torch.log(probabilities[0, label] + 1e-12).item())
                context = EventContext(
                    event_index=event_index,
                    src_node=int(src_cpu.item()),
                    dst_node=int(dst_cpu.item()),
                    timestamp=int(t_cpu.item()),
                    label=int(label),
                    memory_inputs=_store_tensor(memory_inputs),
                    last_update=_store_tensor(last_update),
                    edge_index=_store_tensor(edge_index),
                    edge_messages=_store_tensor(edge_messages),
                    edge_times=_store_tensor(edge_times),
                    base_embeddings=_store_tensor(embeddings),
                    logits=_store_tensor(logits),
                    probabilities=_store_tensor(probabilities),
                    prob_label=prob_label,
                    loss=loss,
                    src_local_index=int(model.assoc[src].item()),
                    dst_local_index=int(model.assoc[dst].item()),
                    node_ids=_store_tensor(n_id),
                    raw_message=_store_tensor(msg_cpu[0]),
                )
                return context

            memory.update_state(src, dst, t, msg)
            loader.insert(src, dst)
            processed += 1

    raise IndexError(f"Event index {event_index} out of range (processed {processed}).")


def stream_event_contexts(
    data: TemporalData,
    memory: torch.nn.Module,
    gnn: torch.nn.Module,
    link_pred: torch.nn.Module,
    *,
    device: Optional[torch.device] = None,
):
    """Yields EventContext objects for every event while streaming once."""
    device = device or model.device
    memory.reset_state()
    loader = _init_neighbor_loader(device)
    loader.reset_state()

    temporal_loader = TemporalDataLoader(data, batch_size=1, shuffle=False)

    with torch.no_grad():
        for event_index, batch in enumerate(temporal_loader):
            src_cpu = batch.src
            dst_cpu = batch.dst
            t_cpu = batch.t
            msg_cpu = batch.msg

            src = src_cpu.to(device)
            dst = dst_cpu.to(device)
            t = t_cpu.to(device)
            msg = msg_cpu.to(device)

            n_id = torch.cat([src, dst]).unique()
            n_id, edge_index, e_id = loader(n_id)
            model.assoc[n_id] = torch.arange(n_id.size(0), device=device)

            memory_inputs, last_update = memory(n_id)
            e_id_cpu = e_id.cpu()
            edge_times = data.t[e_id_cpu].to(device)
            edge_messages = data.msg[e_id_cpu].to(device)

            embeddings = gnn(memory_inputs, last_update, edge_index, edge_times, edge_messages)
            logits = link_pred(embeddings[model.assoc[src]], embeddings[model.assoc[dst]])
            probabilities = logits.softmax(dim=-1)

            label = tensor_find(msg_cpu[0][node_embedding_dim:-node_embedding_dim], 1) - 1
            prob_label = probabilities[0, label].item()
            loss = -float(torch.log(probabilities[0, label] + 1e-12).item())

            context = EventContext(
                event_index=event_index,
                src_node=int(src_cpu.item()),
                dst_node=int(dst_cpu.item()),
                timestamp=int(t_cpu.item()),
                label=int(label),
                memory_inputs=_store_tensor(memory_inputs),
                last_update=_store_tensor(last_update),
                edge_index=_store_tensor(edge_index),
                edge_messages=_store_tensor(edge_messages),
                edge_times=_store_tensor(edge_times),
                base_embeddings=_store_tensor(embeddings),
                logits=_store_tensor(logits),
                probabilities=_store_tensor(probabilities),
                prob_label=prob_label,
                loss=loss,
                src_local_index=int(model.assoc[src].item()),
                dst_local_index=int(model.assoc[dst].item()),
                node_ids=_store_tensor(n_id),
                raw_message=_store_tensor(msg_cpu[0]),
            )

            yield context

            memory.update_state(src, dst, t, msg)
            loader.insert(src, dst)


def compute_event_contexts(
    data: TemporalData,
    memory: torch.nn.Module,
    gnn: torch.nn.Module,
    link_pred: torch.nn.Module,
    indices: Iterable[int],
    device: Optional[torch.device] = None,
) -> Dict[int, EventContext]:
    """Materialize contexts for the requested indices."""
    device = device or model.device
    requested = set(int(i) for i in indices)
    contexts: Dict[int, EventContext] = {}
    if not requested:
        return contexts

    for context in stream_event_contexts(data, memory, gnn, link_pred, device=device):
        if context.event_index in requested:
            contexts[context.event_index] = context
            if len(contexts) == len(requested):
                break
    return contexts


def collect_all_event_contexts(
    data: TemporalData,
    memory: torch.nn.Module,
    gnn: torch.nn.Module,
    link_pred: torch.nn.Module,
    *,
    device: Optional[torch.device] = None,
) -> Dict[int, EventContext]:
    """Collects contexts for every event in the stream."""
    contexts: Dict[int, EventContext] = {}
    for context in stream_event_contexts(data, memory, gnn, link_pred, device=device):
        contexts[context.event_index] = context
    return contexts


def collect_contexts(
    data: TemporalData,
    memory: torch.nn.Module,
    gnn: torch.nn.Module,
    link_pred: torch.nn.Module,
    *,
    predicate=None,
    limit: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[int, EventContext]:
    """Collects contexts satisfying `predicate` until `limit`."""
    contexts: Dict[int, EventContext] = {}

    def should_include(ctx: EventContext) -> bool:
        return True if predicate is None else bool(predicate(ctx))

    for context in stream_event_contexts(data, memory, gnn, link_pred, device=device):
        if should_include(context):
            contexts[context.event_index] = context
            if limit is not None and len(contexts) >= limit:
                break
    return contexts


def _timestamp_str_to_ns(text: str) -> Optional[int]:
    text = text.strip()
    if not text:
        return None
    if "." in text:
        base, frac = text.split(".", 1)
    else:
        base, frac = text, ""
    try:
        base_ns = datetime_to_ns_time_US(base)
    except ValueError:
        return None
    digits = "".join(ch for ch in frac if ch.isdigit())
    if digits:
        digits = (digits + "000000000")[:9]
        base_ns += int(digits)
    return base_ns


def load_attack_intervals() -> List[Tuple[int, int, str]]:
    """Returns (start_ns, end_ns, path) for known attack windows."""
    intervals: List[Tuple[int, int, str]] = []
    for path in fetch_attack_list():
        name = os.path.basename(path).replace(".txt", "")
        if "~" not in name:
            continue
        start_raw, end_raw = name.split("~", 1)
        start_ns = _timestamp_str_to_ns(start_raw)
        end_ns = _timestamp_str_to_ns(end_raw)
        if start_ns is None or end_ns is None:
            continue
        intervals.append((start_ns, end_ns, path))
    intervals.sort(key=lambda x: x[0])
    return intervals
