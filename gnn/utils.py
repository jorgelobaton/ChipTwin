"""
Shared utilities for the GNN world-model module.

Provides graph construction helpers, a pure-PyTorch ``scatter_add``
implementation (no PyTorch Geometric dependency), and checkpoint I/O.
"""

from __future__ import annotations

import math
import os
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
# scatter_add  (replaces torch_scatter / PyG dependency)
# ──────────────────────────────────────────────────────────────

def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Pure-PyTorch scatter-add along *dim*.

    Parameters
    ----------
    src : Tensor of shape ``(E, *)``
        Source values.
    index : LongTensor of shape ``(E,)``
        Target indices (same leading dim as *src*).
    dim : int
        Dimension along which to scatter.
    dim_size : int, optional
        Size of the output along *dim*.  Inferred from ``index.max()+1``
        when not given.

    Returns
    -------
    Tensor of shape ``(dim_size, *)``
    """
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    idx = index.unsqueeze(-1).expand_as(src)
    out.scatter_add_(dim, idx, src)
    return out


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    """Pure-PyTorch scatter-mean along *dim*."""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1
    summed = scatter_add(src, index, dim=dim, dim_size=dim_size)
    count = scatter_add(
        torch.ones(src.shape[0], 1, device=src.device, dtype=src.dtype),
        index,
        dim=dim,
        dim_size=dim_size,
    )
    return summed / count.clamp(min=1.0)


# ──────────────────────────────────────────────────────────────
# Graph construction from simulator state
# ──────────────────────────────────────────────────────────────

def build_graph_from_sim_state(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    edge_indices: torch.Tensor,
    rest_lengths: torch.Tensor,
    spring_Y_log: torch.Tensor,
    yield_strain: torch.Tensor,
    hardening_factor: torch.Tensor,
    break_strain: Optional[torch.Tensor],
    is_controller: torch.Tensor,
    enable_breakage: bool = False,
) -> Dict[str, torch.Tensor]:
    """Build the node/edge feature tensors expected by :class:`PhysicsGNN`.

    All inputs are expected on the same device and as ``float32`` tensors
    (except ``edge_indices`` which is ``long``).

    Parameters
    ----------
    positions : (N, 3)
    velocities : (N, 3)
    edge_indices : (E, 2)  – spring endpoint indices
    rest_lengths : (E,)
    spring_Y_log : (E,)  – log-space stiffness per spring
    yield_strain : (E,)
    hardening_factor : (E,)
    break_strain : (E,) or ``None``
    is_controller : (N,)  – 1.0 for controller nodes, 0.0 otherwise
    enable_breakage : bool

    Returns
    -------
    dict with keys:
        ``node_features``  (N, node_dim)
        ``edge_features``  (E, edge_dim)
        ``edge_index``     (2, E)  – COO format (src, dst)
        ``positions``      (N, 3)
        ``velocities``     (N, 3)
    """
    device = positions.device
    N = positions.shape[0]
    E = edge_indices.shape[0]

    src = edge_indices[:, 0].long()
    dst = edge_indices[:, 1].long()

    # Filter out any edges that reference nodes beyond N (e.g. controller nodes)
    valid = (src < N) & (dst < N) & (src >= 0) & (dst >= 0)
    if not valid.all():
        valid_idx = valid.nonzero(as_tuple=True)[0]
        src = src[valid_idx]
        dst = dst[valid_idx]
        rest_lengths = rest_lengths[valid_idx]
        spring_Y_log = spring_Y_log[valid_idx]
        yield_strain = yield_strain[valid_idx]
        hardening_factor = hardening_factor[valid_idx]
        if break_strain is not None:
            break_strain = break_strain[valid_idx]
        E = src.shape[0]

    # ── per-node mean physical params (averaged from incident springs) ──
    node_mean_yield = scatter_mean(yield_strain.unsqueeze(-1), src, dim=0, dim_size=N).squeeze(-1)
    node_mean_hard = scatter_mean(hardening_factor.unsqueeze(-1), src, dim=0, dim_size=N).squeeze(-1)
    node_mean_stiff = scatter_mean(spring_Y_log.unsqueeze(-1), src, dim=0, dim_size=N).squeeze(-1)
    # also average from dst side
    node_mean_yield = (node_mean_yield + scatter_mean(yield_strain.unsqueeze(-1), dst, dim=0, dim_size=N).squeeze(-1)) / 2.0
    node_mean_hard = (node_mean_hard + scatter_mean(hardening_factor.unsqueeze(-1), dst, dim=0, dim_size=N).squeeze(-1)) / 2.0
    node_mean_stiff = (node_mean_stiff + scatter_mean(spring_Y_log.unsqueeze(-1), dst, dim=0, dim_size=N).squeeze(-1)) / 2.0

    node_features = torch.stack([
        node_mean_yield,
        node_mean_hard,
        node_mean_stiff,
        is_controller.float(),
    ], dim=-1)  # (N, 4)
    node_features = torch.cat([positions, velocities, node_features], dim=-1)  # (N, 10)

    # ── edge features ──
    x_i = positions[src]
    x_j = positions[dst]
    diff = x_j - x_i
    current_length = diff.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    unit_dir = diff / current_length
    strain = (current_length.squeeze(-1) / rest_lengths.clamp(min=1e-8)) - 1.0

    edge_feat_list = [
        rest_lengths.unsqueeze(-1),
        current_length,
        unit_dir,                   # (E, 3)
        strain.unsqueeze(-1),
        spring_Y_log.unsqueeze(-1),
        yield_strain.unsqueeze(-1),
        hardening_factor.unsqueeze(-1),
    ]
    if enable_breakage and break_strain is not None:
        edge_feat_list.append(break_strain.unsqueeze(-1))

    edge_features = torch.cat(edge_feat_list, dim=-1)  # (E, 10 or 11)

    # COO format (2, E), with both directions
    edge_index = torch.stack([src, dst], dim=0)

    return {
        "node_features": node_features,
        "edge_features": edge_features,
        "edge_index": edge_index,
        "positions": positions,
        "velocities": velocities,
    }


# ──────────────────────────────────────────────────────────────
# Checkpoint I/O
# ──────────────────────────────────────────────────────────────

def save_gnn_checkpoint(
    model: torch.nn.Module,
    path: str,
    epoch: int = 0,
    val_loss: float = float("inf"),
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a GNN model checkpoint.

    Parameters
    ----------
    model : nn.Module
    path : str – file path (directories will be created).
    epoch : int
    val_loss : float
    extra : dict, optional – any additional metadata to store.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
    }
    if extra is not None:
        state.update(extra)
    torch.save(state, path)


def load_gnn_checkpoint(
    model: torch.nn.Module,
    path: str,
    device: str = "cuda:0",
) -> Dict[str, Any]:
    """Load a GNN checkpoint into *model* and return the metadata dict.

    Parameters
    ----------
    model : nn.Module – will be updated **in-place**.
    path : str
    device : str

    Returns
    -------
    dict – the full checkpoint dictionary.
    """
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt
