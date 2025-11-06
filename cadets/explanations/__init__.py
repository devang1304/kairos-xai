"""
Utilities for generating explanations of the Kairos TGN model.

This package bundles lightweight wrappers around PyG's GNNExplainer and
PGExplainer, along with common utilities and evaluation metrics.
"""

from . import utils, gnn_explainer, pg_explainer, metrics

__all__ = ["utils", "gnn_explainer", "pg_explainer", "metrics"]
