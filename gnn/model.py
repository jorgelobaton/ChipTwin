"""
Physics-informed Graph Neural Network for the ChipTwin simulator.

Implements an Encode–Process–Decode architecture following Pfaff et al.
"Learning Mesh-Based Simulation with Graph Networks" (ICLR 2021).

The model is fully differentiable with respect to the per-edge physical
parameters (``spring_Y_log``, ``yield_strain``, ``hardening_factor``,
``break_strain``) so that online fine-tuning can back-propagate through
the network to update only those parameters while keeping GNN weights
frozen.

No PyTorch Geometric dependency – message passing is implemented manually
with ``scatter_add``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .utils import scatter_add


# ──────────────────────────────────────────────────────────────
# Small building block: 2-layer MLP with LayerNorm
# ──────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """Two-hidden-layer MLP with ReLU and optional LayerNorm on output.

    Parameters
    ----------
    in_dim : int
    hidden_dim : int
    out_dim : int
    layer_norm : bool – apply LayerNorm to the output.
    """

    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, layer_norm: bool = True
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )
        self.ln = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.ln(self.net(x))


# ──────────────────────────────────────────────────────────────
# Processor – single message-passing step
# ──────────────────────────────────────────────────────────────

class ProcessorStep(nn.Module):
    """One message-passing step with residual connections.

    Edge update:  ``e_ij' = MLP(concat(h_i, h_j, e_ij))``
    Node update:  ``h_i'  = MLP(concat(h_i, mean(e_ij')))``

    Both updates use residual connections.

    Parameters
    ----------
    node_dim : int – node embedding dimensionality.
    edge_dim : int – edge embedding dimensionality.
    hidden_dim : int – hidden size for MLPs.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.edge_mlp = MLP(node_dim * 2 + edge_dim, hidden_dim, edge_dim, layer_norm=True)
        self.node_mlp = MLP(node_dim + edge_dim, hidden_dim, node_dim, layer_norm=True)

    def forward(
        self,
        node_emb: torch.Tensor,
        edge_emb: torch.Tensor,
        src: torch.LongTensor,
        dst: torch.LongTensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one message-passing step.

        Parameters
        ----------
        node_emb : (N, node_dim)
        edge_emb : (E, edge_dim)
        src, dst : (E,)  – edge endpoint indices (COO).
        num_nodes : int

        Returns
        -------
        (node_emb', edge_emb')
        """
        # Edge update
        h_i = node_emb[src]
        h_j = node_emb[dst]
        edge_input = torch.cat([h_i, h_j, edge_emb], dim=-1)
        edge_emb_new = self.edge_mlp(edge_input) + edge_emb  # residual

        # Aggregate updated edges → target nodes (mean)
        agg = scatter_add(edge_emb_new, dst, dim=0, dim_size=num_nodes)
        count = scatter_add(
            torch.ones(dst.shape[0], 1, device=dst.device, dtype=node_emb.dtype),
            dst, dim=0, dim_size=num_nodes,
        )
        agg = agg / count.clamp(min=1.0)

        # Node update
        node_input = torch.cat([node_emb, agg], dim=-1)
        node_emb_new = self.node_mlp(node_input) + node_emb  # residual

        return node_emb_new, edge_emb_new


# ──────────────────────────────────────────────────────────────
# Full GNN model
# ──────────────────────────────────────────────────────────────

class PhysicsGNN(nn.Module):
    """Encode–Process–Decode GNN for physics-based simulation.

    NODE features (input_node_dim = 10):
        position (3), velocity (3), mean_yield_strain (1),
        mean_hardening_factor (1), mean_log_stiffness (1),
        is_controller (1).

    EDGE features (input_edge_dim = 9 or 10 with breakage):
        rest_length (1), current_length (1), unit_direction (3),
        strain (1), spring_Y_log (1), yield_strain (1),
        hardening_factor (1), [break_strain (1)].

    OUTPUT:
        Per-node predicted velocity delta (3D).

    Parameters
    ----------
    input_node_dim : int
    input_edge_dim : int
    hidden_dim : int – latent embedding size (default 128).
    message_passing_steps : int – number of processor iterations (default 10).
    """

    def __init__(
        self,
        input_node_dim: int = 10,
        input_edge_dim: int = 9,
        hidden_dim: int = 128,
        message_passing_steps: int = 10,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_passing_steps = message_passing_steps

        # Encoder MLPs
        self.node_encoder = MLP(input_node_dim, hidden_dim, hidden_dim, layer_norm=True)
        self.edge_encoder = MLP(input_edge_dim, hidden_dim, hidden_dim, layer_norm=True)

        # Processor – stack of message-passing steps
        self.processors = nn.ModuleList([
            ProcessorStep(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(message_passing_steps)
        ])

        # Decoder: node embedding → velocity delta (3D)
        self.decoder = MLP(hidden_dim, hidden_dim, 3, layer_norm=False)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        edge_index: torch.LongTensor,
    ) -> torch.Tensor:
        """Forward pass – predict per-node velocity delta.

        Parameters
        ----------
        node_features : (N, input_node_dim)
        edge_features : (E, input_edge_dim)
        edge_index : (2, E)  – (src, dst) in COO format

        Returns
        -------
        delta_v : (N, 3)
        """
        src, dst = edge_index[0], edge_index[1]
        num_nodes = node_features.shape[0]

        # Encode
        node_emb = self.node_encoder(node_features)
        edge_emb = self.edge_encoder(edge_features)

        # Process (gradient checkpointing to save GPU memory)
        for processor in self.processors:
            if self.training:
                node_emb, edge_emb = grad_checkpoint(
                    processor, node_emb, edge_emb, src, dst, num_nodes,
                    use_reentrant=False,
                )
            else:
                node_emb, edge_emb = processor(node_emb, edge_emb, src, dst, num_nodes)

        # Decode
        delta_v = self.decoder(node_emb)
        return delta_v

    def rollout_step(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        edge_index: torch.LongTensor,
        rest_lengths: torch.Tensor,
        spring_Y_log: torch.Tensor,
        yield_strain: torch.Tensor,
        hardening_factor: torch.Tensor,
        break_strain: Optional[torch.Tensor],
        is_controller: torch.Tensor,
        dt_frame: float,
        enable_breakage: bool = False,
        controller_mask: Optional[torch.Tensor] = None,
        controller_target_pos: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One-step rollout: predict delta_v, integrate to get next state.

        Parameters
        ----------
        positions, velocities : (N, 3)
        (remaining parameters define the graph features – see ``build_graph_from_sim_state``.)
        dt_frame : float – time step for position integration (1/FPS).
        controller_mask : (N,) bool – True for controller nodes.
        controller_target_pos : (N_ctrl, 3) – known next-frame positions
            for controller nodes.  After Euler integration, controller
            node positions are *overridden* with these targets and their
            velocities are recomputed as ``(target - old_pos) / dt``.

        Returns
        -------
        (new_positions, new_velocities) each (N, 3)
        """
        from .utils import build_graph_from_sim_state

        graph = build_graph_from_sim_state(
            positions, velocities, edge_index,
            rest_lengths, spring_Y_log, yield_strain, hardening_factor,
            break_strain, is_controller, enable_breakage,
        )

        delta_v = self.forward(
            graph["node_features"],
            graph["edge_features"],
            graph["edge_index"],
        )

        new_vel = velocities + delta_v
        new_pos = positions + new_vel * dt_frame

        # Override controller nodes with known targets
        if controller_mask is not None and controller_target_pos is not None:
            old_ctrl_pos = positions[controller_mask]
            new_pos = new_pos.clone()
            new_vel = new_vel.clone()
            new_pos[controller_mask] = controller_target_pos
            new_vel[controller_mask] = (controller_target_pos - old_ctrl_pos) / dt_frame

        return new_pos, new_vel
