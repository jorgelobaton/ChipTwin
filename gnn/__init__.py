"""
GNN World Model extension for the ChipTwin physics simulator.

This module provides an optional Graph Neural Network (GNN) that can replace
the expensive Warp spring-mass simulator at inference time and support rapid
online fine-tuning of physical properties for unseen chip instances.

All functionality is gated behind the ``use_gnn_world_model`` flag in
``configs/real.yaml``.  When disabled (the default), the original ChipTwin
pipeline operates unchanged.
"""

from .model import PhysicsGNN
from .utils import (
    build_graph_from_sim_state,
    scatter_add,
    load_gnn_checkpoint,
    save_gnn_checkpoint,
)

__all__ = [
    "PhysicsGNN",
    "build_graph_from_sim_state",
    "scatter_add",
    "load_gnn_checkpoint",
    "save_gnn_checkpoint",
]
