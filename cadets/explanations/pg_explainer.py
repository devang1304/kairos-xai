import time
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch_geometric.data import TemporalData
from torch_geometric.explain import ExplainerConfig, ModelConfig
from torch_geometric.explain.algorithm import PGExplainer as PGExplainerAlgo

try:
    from .. import model
except ImportError:  # pragma: no cover
    import model
from . import metrics, utils

PG_EPOCHS = 30
PG_LR = 0.003
MASK_THRESHOLD = 0.5


def train_pg_explainer(
    memory: torch.nn.Module,
    gnn: torch.nn.Module,
    link_pred: torch.nn.Module,
    data: TemporalData,
    train_indices: Iterable[int],
    *,
    contexts: Optional[Dict[int, utils.EventContext]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[PGExplainerAlgo, Dict[int, utils.EventContext]]:
    """Trains a PGExplainer using the provided event indices."""
    device = device or model.device
    if contexts is None:
        contexts = utils.compute_event_contexts(data, memory, gnn, link_pred, train_indices, device=device)

    algorithm = PGExplainerAlgo(epochs=PG_EPOCHS, lr=PG_LR)
    algorithm.model_config = ModelConfig(
        mode="multiclass_classification",
        task_level="edge",
        return_type="logits",
    )
    algorithm.explainer_config = ExplainerConfig(
        explanation_type="phenomenon",
        node_mask_type=None,
        edge_mask_type="object",
    )

    for epoch in range(PG_EPOCHS):
        for idx in train_indices:
            context = contexts[idx]
            wrapper = utils.TemporalLinkWrapper(gnn, link_pred, context, device)
            target = torch.tensor([max(context.label, 0)], device=device)
            algorithm.train(
                epoch,
                wrapper,
                context.memory_inputs.to(device),
                context.edge_index.to(device),
                target=target,
                edge_attr=context.edge_messages.to(device),
                edge_t=context.edge_times.to(device),
            )

    return algorithm, contexts


def explain_event(
    context: utils.EventContext,
    gnn: torch.nn.Module,
    link_pred: torch.nn.Module,
    algorithm: PGExplainerAlgo,
    device: torch.device,
) -> Dict[str, float]:
    """Runs PGExplainer on a single event."""
    wrapper = utils.TemporalLinkWrapper(gnn, link_pred, context, device)
    target = torch.tensor([max(context.label, 0)], device=device)

    start = time.perf_counter()
    explanation = algorithm(
        wrapper,
        context.memory_inputs.to(device),
        context.edge_index.to(device),
        target=target,
        edge_attr=context.edge_messages.to(device),
        edge_t=context.edge_times.to(device),
    )
    runtime = time.perf_counter() - start

    edge_mask = explanation.edge_mask.detach()
    return metrics.evaluate_mask(wrapper, context, edge_mask, threshold=MASK_THRESHOLD, runtime=runtime)
