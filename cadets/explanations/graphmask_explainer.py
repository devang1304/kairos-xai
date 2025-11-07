"""
GraphMask-style temporal explanations for the Kairos TGN model.

This implementation adapts the GraphMask idea (learning sparse edge masks
that preserve the model prediction) to the EventContext abstraction used in
Kairos.  For each temporal event we optimise a sigmoid gate per historical
edge and penalise dense masks.  Running it across multiple events and
aggregating the scores yields a graph-level story for the attack window.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from .utils import EventContext, TemporalLinkWrapper
try:  # pragma: no cover
    from ..config import include_edge_type as _DEFAULT_INCLUDE_EDGE_TYPE, node_embedding_dim as _DEFAULT_NODE_EMBEDDING_DIM
except ImportError:  # pragma: no cover
    try:
        from config import include_edge_type as _DEFAULT_INCLUDE_EDGE_TYPE, node_embedding_dim as _DEFAULT_NODE_EMBEDDING_DIM  # type: ignore
    except ImportError:  # pragma: no cover
        _DEFAULT_INCLUDE_EDGE_TYPE = None
        _DEFAULT_NODE_EMBEDDING_DIM = None


@dataclass
class GraphMaskResult:
    event_index: int
    edge_importance: Tensor  # shape: (num_edges,)
    edges: List[Tuple[int, int, str, int]]
    loss_history: List[float]


class GraphMaskExplainer:
    """
    Lightweight GraphMask-style explainer tailored for EventContext objects.

    Parameters
    ----------
    sparsity_weight: float
        Coefficient encouraging sparse masks (default 1e-3).
    entropy_weight: float
        Coefficient discouraging ambiguous masks (default 1e-3).
    epochs: int
        Number of optimisation steps per event (default 200).
    lr: float
        Learning rate for Adam optimiser (default 0.01).
    """

    def __init__(
        self,
        sparsity_weight: float = 1e-3,
        entropy_weight: float = 1e-3,
        epochs: int = 200,
        lr: float = 0.01,
        include_edge_type=None,
        node_embedding_dim=None,
    ) -> None:
        self.sparsity_weight = sparsity_weight
        self.entropy_weight = entropy_weight
        self.epochs = epochs
        self.lr = lr
        self.include_edge_type = include_edge_type or _DEFAULT_INCLUDE_EDGE_TYPE
        self.node_embedding_dim = node_embedding_dim or _DEFAULT_NODE_EMBEDDING_DIM
        if self.include_edge_type is None or self.node_embedding_dim is None:
            raise ValueError(
                "GraphMaskExplainer requires 'include_edge_type' and 'node_embedding_dim'. "
                "Pass them to the constructor or ensure cadets.config is importable."
            )

    def explain_event(
        self,
        context: EventContext,
        wrapper: TemporalLinkWrapper,
        device: torch.device,
    ) -> GraphMaskResult:
        """
        Learn an edge mask for a single temporal event.
        """
        edge_messages = context.edge_messages.to(device)
        edge_index = context.edge_index.to(device)
        edge_times = context.edge_times.to(device)

        num_edges = edge_messages.size(0)
        if num_edges == 0:
            return GraphMaskResult(
                event_index=context.event_index,
                edge_importance=torch.tensor([]),
                edges=[],
                loss_history=[],
            )

        alpha = torch.zeros(num_edges, device=device, requires_grad=True)
        optimizer = Adam([alpha], lr=self.lr)

        target = torch.tensor([max(context.label, 0)], device=device)

        losses: List[float] = []

        for _ in range(self.epochs):
            optimizer.zero_grad()
            mask = torch.sigmoid(alpha)  # (num_edges,)
            masked_messages = edge_messages * mask.unsqueeze(-1)
            logits = wrapper(
                context.memory_inputs.to(device),
                edge_index,
                edge_attr=masked_messages,
                edge_t=edge_times,
            )
            ce = F.cross_entropy(logits, target)

            sparsity = mask.mean()
            entropy = -(mask * torch.log(mask + 1e-8) + (1 - mask) * torch.log(1 - mask + 1e-8)).mean()

            loss = ce + self.sparsity_weight * sparsity + self.entropy_weight * entropy
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        importance = torch.sigmoid(alpha).detach().cpu()
        edge_meta: List[Tuple[int, int, str, int]] = []
        for idx in range(num_edges):
            src = int(context.edge_index[0, idx])
            dst = int(context.edge_index[1, idx])
            timestamp = int(context.edge_times[idx].item())
            msg_slice = context.edge_messages[idx]
            relation_idx = torch.argmax(msg_slice[self.node_embedding_dim:-self.node_embedding_dim]).item()
            relation = self.include_edge_type[relation_idx]
            edge_meta.append((src, dst, relation, timestamp))

        return GraphMaskResult(
            event_index=context.event_index,
            edge_importance=importance,
            edges=edge_meta,
            loss_history=losses,
        )

    def explain_window(
        self,
        contexts: Iterable[Tuple[int, EventContext]],
        wrapper_factory,
        device: torch.device,
        top_k_events: int | None = None,
    ) -> List[GraphMaskResult]:
        """
        Run GraphMask on a selection of events and aggregate the results.
        """
        ordered_contexts = list(contexts)
        if top_k_events is not None:
            ordered_contexts = ordered_contexts[:top_k_events]

        results: List[GraphMaskResult] = []
        for _, context in ordered_contexts:
            wrapper = wrapper_factory(context)
            results.append(self.explain_event(context, wrapper, device))
        return results

    @staticmethod
    def aggregate(
        results: Iterable[GraphMaskResult],
    ) -> Dict[Tuple[int, int, str], Dict[str, List[float] | float | int | List[int]]]:
        """
        Aggregate per-event masks into a single mapping keyed by edge identity.
        """
        scores: Dict[Tuple[int, int, str], List[float]] = {}
        timestamps: Dict[Tuple[int, int, str], List[int]] = {}
        for res in results:
            for (src, dst, relation, ts), weight in zip(res.edges, res.edge_importance.tolist()):
                key = (src, dst, relation)
                scores.setdefault(key, []).append(weight)
                timestamps.setdefault(key, []).append(ts)

        aggregated: Dict[Tuple[int, int, str], Dict[str, List[float] | float | int | List[int]]] = {}
        for key, weights in scores.items():
            aggregated[key] = {
                "weight": float(sum(weights) / len(weights)),
                "count": len(weights),
                "timestamps": sorted(timestamps.get(key, [])),
            }
        return aggregated
