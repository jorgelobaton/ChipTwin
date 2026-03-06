"""
Online fine-tuning of per-spring physical parameters through a frozen GNN.

Given a pre-trained GNN (weights frozen) and live observations of an unseen
chip instance, this script identifies the chip's physical parameters by
back-propagating the observation loss **only** through the per-edge physical
parameter tensors (spring_Y_log, yield_strain, hardening_factor, break_strain).

After fine-tuning, a full GNN rollout is executed with the identified
parameters and saved to ``inference_gnn_finetuned.pkl``.

Usage
-----
    python gnn/online_finetune.py --config configs/real.yaml \\
        --base_path ./data/different_types --case_name demo_cable \\
        --use_gnn_world_model true --gnn_online_finetune true \\
        --gnn_checkpoint_dir checkpoints/gnn/ --gnn_finetune_frames 10 \\
        --gnn_finetune_lr 1e-3 --gnn_finetune_steps 20
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg, logger

try:
    import warp as wp
except ImportError:
    wp = None

try:
    from pytorch3d.loss import chamfer_distance as _chamfer_distance
    _HAS_PYTORCH3D = True
except ImportError:
    _HAS_PYTORCH3D = False

from gnn.model import PhysicsGNN
from gnn.utils import build_graph_from_sim_state, load_gnn_checkpoint


# ──────────────────────────────────────────────────────────────
# Loss helpers
# ──────────────────────────────────────────────────────────────

def _chamfer_loss(
    pred_positions: torch.Tensor,
    gt_points: torch.Tensor,
    gt_vis: torch.Tensor,
    num_surface_points: int,
) -> torch.Tensor:
    """One-directional Chamfer loss (GT visible → predicted surface).

    Parameters
    ----------
    pred_positions : (N, 3) – all predicted node positions.
    gt_points : (M, 3) – GT observed points for this frame.
    gt_vis : (M,) bool – visibility mask.
    num_surface_points : int

    Returns
    -------
    Scalar loss.
    """
    visible = gt_points[gt_vis.bool()]
    if visible.shape[0] == 0:
        return torch.tensor(0.0, device=pred_positions.device)
    pred_surface = pred_positions[:num_surface_points]

    if _HAS_PYTORCH3D:
        loss, _ = _chamfer_distance(
            visible.unsqueeze(0), pred_surface.unsqueeze(0),
            single_directional=True, norm=1,
        )
        return loss
    else:
        # Fallback: simple nearest-neighbour L1
        dists = torch.cdist(visible, pred_surface, p=1)
        return dists.min(dim=1).values.mean()


def _tracking_loss(
    pred_positions: torch.Tensor,
    gt_track: torch.Tensor,
    track_mask: torch.Tensor,
    track_idx: torch.LongTensor,
) -> torch.Tensor:
    """Smooth L1 tracking loss for tracked points.

    Parameters
    ----------
    pred_positions : (N, 3)
    gt_track : (T, 3) – GT tracked point positions (may have NaN).
    track_mask : (T,) bool
    track_idx : (T,) long – corresponding indices into pred_positions.

    Returns
    -------
    Scalar loss.
    """
    valid = track_mask.bool() & (~torch.isnan(gt_track).any(dim=-1))
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred_positions.device)
    return F.smooth_l1_loss(pred_positions[track_idx[valid]], gt_track[valid])


# ──────────────────────────────────────────────────────────────
# Clamping
# ──────────────────────────────────────────────────────────────

def _clamp_params(
    spring_Y_log: torch.Tensor,
    yield_strain: torch.Tensor,
    hardening_factor: torch.Tensor,
    break_strain: Optional[torch.Tensor],
) -> None:
    """In-place clamp physical parameters to valid ranges."""
    with torch.no_grad():
        spring_Y_log.clamp_(min=0.0)  # at least log(1)
        yield_strain.clamp_(min=0.001, max=1.0)
        hardening_factor.clamp_(min=0.0, max=1.0)
        if break_strain is not None:
            break_strain.clamp_(min=0.05, max=2.0)


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def run_gnn_online_finetune(args: argparse.Namespace) -> None:
    """Online identification of physical parameters for an unseen chip.

    Procedure
    ---------
    1. Load frozen GNN (weights NOT updated).
    2. Initialise per-spring physical parameters from config / CMA-ES.
    3. For each incoming frame (up to ``gnn_finetune_frames``):
       a. GNN forward → observation loss (Chamfer + tracking).
       b. Adam steps on physical params only.
       c. Clamp to valid ranges.
    4. Full GNN rollout with identified params → ``inference_gnn_finetuned.pkl``.
    5. Save identified params to JSON.

    Parameters
    ----------
    args : Namespace
    """
    base_path = args.base_path
    case_name = args.case_name
    device = getattr(cfg, "device", "cuda:0")
    enable_breakage = getattr(cfg, "enable_breakage", False)

    suffix = ""
    if getattr(args, "enable_plasticity", False):
        suffix += "_ep"
    if getattr(args, "enable_breakage", False):
        suffix += "_brk"
    base_dir = f"experiments/{case_name}{suffix}"

    # Load CMA-ES optimal params
    optimal_path = f"experiments_optimization/{case_name}{suffix}/optimal_params.pkl"
    if os.path.exists(optimal_path):
        with open(optimal_path, "rb") as f:
            optimal_params = pickle.load(f)
        cfg.set_optimal_params(optimal_params)

    # Calibration
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array([np.linalg.inv(c) for c in c2ws])
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        meta = json.load(f)
    cfg.intrinsics = np.array(meta["intrinsics"])
    cfg.WH = meta["WH"]
    cfg.overlay_path = f"{base_path}/{case_name}/color"
    if "camera_ids" in meta:
        cfg.camera_ids = meta["camera_ids"]

    if getattr(args, "enable_plasticity", False):
        cfg.enable_plasticity = True
    if getattr(args, "enable_breakage", False):
        cfg.enable_breakage = True

    logger.set_log_file(path=base_dir, name="gnn_finetune_log")

    # Build trainer (for init graph + GT data)
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )
    sim = trainer.simulator
    n_springs = sim.n_springs

    # ── Initialise physical parameters as leaf tensors ──
    # Start from CMA-ES / checkpoint defaults
    ckpt_files = glob.glob(f"{base_dir}/train/best_*.pth")
    if len(ckpt_files) > 0:
        ckpt = torch.load(ckpt_files[0], map_location=device)
        spring_Y_log = torch.log(ckpt["spring_Y"]).to(device).float().clone().detach().requires_grad_(True)
        ys_init = ckpt.get("yield_strain")
        if ys_init is not None:
            if not isinstance(ys_init, torch.Tensor):
                ys_init = torch.full((n_springs,), float(ys_init), device=device)
            ys = ys_init.to(device).float().clone().detach().requires_grad_(True)
        else:
            ys = torch.full((n_springs,), cfg.yield_strain, device=device, requires_grad=True)
        hf_init = ckpt.get("hardening_factor")
        if hf_init is not None:
            if not isinstance(hf_init, torch.Tensor):
                hf_init = torch.full((n_springs,), float(hf_init), device=device)
            hf = hf_init.to(device).float().clone().detach().requires_grad_(True)
        else:
            hf = torch.full((n_springs,), cfg.hardening_factor, device=device, requires_grad=True)
    else:
        spring_Y_log = (
            torch.full((n_springs,), float(np.log(cfg.init_spring_Y)), device=device)
            .requires_grad_(True)
        )
        ys = torch.full((n_springs,), cfg.yield_strain, device=device, requires_grad=True)
        hf = torch.full((n_springs,), cfg.hardening_factor, device=device, requires_grad=True)

    bs = None
    if enable_breakage:
        bs = torch.full((n_springs,), cfg.break_strain, device=device, requires_grad=True)

    opt_params = [spring_Y_log, ys, hf]
    if bs is not None:
        opt_params.append(bs)

    # ── Controller trajectory ──
    N_obj = sim.num_object_points
    has_controller = trainer.controller_points is not None
    if has_controller:
        ctrl_traj = torch.stack(
            [t.float() if isinstance(t, torch.Tensor) else torch.tensor(t, dtype=torch.float32)
             for t in trainer.controller_points], dim=0
        ).to(device)  # (T, N_ctrl, 3)
        N_ctrl = ctrl_traj.shape[1]
    else:
        ctrl_traj = None
        N_ctrl = 0
    N_total = N_obj + N_ctrl

    # Edge / graph structure – ALL springs (object + controller-to-object)
    edge_indices_torch = wp.to_torch(sim.wp_springs, requires_grad=False).long().to(device)
    rest_lengths_torch = wp.to_torch(sim.wp_rest_lengths, requires_grad=False).float().to(device)

    # is_controller flag: 0 for object nodes, 1 for controller nodes
    is_ctrl = torch.zeros(N_total, device=device)
    if has_controller:
        is_ctrl[N_obj:] = 1.0
    ctrl_mask = is_ctrl.bool()  # (N_total,)

    # Load GNN (frozen)
    input_edge_dim = 9 + (1 if enable_breakage else 0)
    model = PhysicsGNN(
        input_node_dim=10,
        input_edge_dim=input_edge_dim,
        hidden_dim=cfg.gnn_hidden_dim,
        message_passing_steps=cfg.gnn_message_passing_steps,
    ).to(device)
    gnn_ckpt_path = os.path.join(cfg.gnn_checkpoint_dir, "best_gnn.pth")
    assert os.path.exists(gnn_ckpt_path), f"GNN checkpoint not found: {gnn_ckpt_path}"
    load_gnn_checkpoint(model, gnn_ckpt_path, device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # GT observation data
    gt_object_points = trainer.object_points  # (F, M, 3)
    gt_vis = trainer.object_visibilities      # (F, M) bool
    num_surface = sim.num_surface_points if sim.num_surface_points else sim.num_object_points

    dt_frame = 1.0 / cfg.FPS
    finetune_frames = cfg.gnn_finetune_frames
    finetune_steps = cfg.gnn_finetune_steps
    finetune_lr = cfg.gnn_finetune_lr

    # Initial state
    # Must call set_init_state to populate wp_states[0] with actual positions
    if sim.enable_plasticity or sim.enable_breakage:
        sim.reset_rest_lengths()
    sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)

    obj_pos = wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False).float().to(device)
    obj_vel = wp.to_torch(sim.wp_states[0].wp_v, requires_grad=False).float().to(device)

    # Build full initial state with controller nodes
    if has_controller:
        ctrl_pos_0 = ctrl_traj[0]
        ctrl_vel_0 = torch.zeros(N_ctrl, 3, device=device)
        pos = torch.cat([obj_pos, ctrl_pos_0], dim=0)
        vel = torch.cat([obj_vel, ctrl_vel_0], dim=0)
    else:
        pos = obj_pos
        vel = obj_vel

    optimizer = torch.optim.Adam(opt_params, lr=finetune_lr)

    logger.info(f"Online fine-tuning: {finetune_frames} frames × {finetune_steps} steps "
                f"(N_obj={N_obj}, N_ctrl={N_ctrl})")

    for frame_idx in range(1, min(finetune_frames + 1, trainer.dataset.frame_len)):
        # Controller target for this frame
        ctrl_target = None
        if has_controller and frame_idx < ctrl_traj.shape[0]:
            ctrl_target = ctrl_traj[frame_idx]  # (N_ctrl, 3)

        # GT for this frame
        gt_pts_frame = torch.tensor(
            gt_object_points[frame_idx], dtype=torch.float32, device=device
        )
        gt_vis_frame = torch.tensor(
            gt_vis[frame_idx], dtype=torch.bool, device=device
        ) if gt_vis is not None else torch.ones(gt_pts_frame.shape[0], dtype=torch.bool, device=device)

        for step in range(finetune_steps):
            optimizer.zero_grad()

            # GNN forward
            pred_pos, pred_vel = model.rollout_step(
                pos.detach(), vel.detach(),
                edge_indices_torch, rest_lengths_torch,
                spring_Y_log, ys, hf, bs, is_ctrl,
                dt_frame, enable_breakage,
                controller_mask=ctrl_mask if has_controller else None,
                controller_target_pos=ctrl_target,
            )

            # Losses – only on object nodes (first N_obj)
            loss_chamfer = _chamfer_loss(pred_pos[:N_obj], gt_pts_frame, gt_vis_frame, num_surface)
            loss = loss_chamfer
            loss.backward()
            optimizer.step()

            _clamp_params(spring_Y_log, ys, hf, bs)

        # Advance state (detach to avoid graph memory buildup)
        with torch.no_grad():
            pos, vel = model.rollout_step(
                pos, vel, edge_indices_torch, rest_lengths_torch,
                spring_Y_log, ys, hf, bs, is_ctrl,
                dt_frame, enable_breakage,
                controller_mask=ctrl_mask if has_controller else None,
                controller_target_pos=ctrl_target,
            )

        logger.info(f"  Frame {frame_idx}: loss={loss.item():.6f}")

    # ── Full rollout with identified params ──
    logger.info("Running full GNN rollout with fine-tuned params …")
    sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)
    obj_pos = wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False).float().to(device)
    obj_vel = wp.to_torch(sim.wp_states[0].wp_v, requires_grad=False).float().to(device)

    if has_controller:
        pos = torch.cat([obj_pos, ctrl_traj[0]], dim=0)
        vel = torch.cat([obj_vel, torch.zeros(N_ctrl, 3, device=device)], dim=0)
    else:
        pos = obj_pos
        vel = obj_vel

    vertices = [pos[:N_obj].cpu()]

    with torch.no_grad():
        for frame_idx in range(1, trainer.dataset.frame_len):
            ctrl_target = None
            if has_controller and frame_idx < ctrl_traj.shape[0]:
                ctrl_target = ctrl_traj[frame_idx]
            pos, vel = model.rollout_step(
                pos, vel, edge_indices_torch, rest_lengths_torch,
                spring_Y_log, ys, hf, bs, is_ctrl,
                dt_frame, enable_breakage,
                controller_mask=ctrl_mask if has_controller else None,
                controller_target_pos=ctrl_target,
            )
            vertices.append(pos[:N_obj].cpu())

    vertices = torch.stack(vertices, dim=0).numpy()  # (T, N_obj, 3)

    save_path = os.path.join(base_dir, "inference_gnn_finetuned.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(vertices, f)
    logger.info(f"Fine-tuned rollout saved → {save_path}")

    # Save identified params to JSON
    params_dict = {
        "spring_Y_log": spring_Y_log.detach().cpu().tolist(),
        "yield_strain": ys.detach().cpu().tolist(),
        "hardening_factor": hf.detach().cpu().tolist(),
    }
    if bs is not None:
        params_dict["break_strain"] = bs.detach().cpu().tolist()
    params_path = os.path.join(base_dir, "gnn_finetuned_params.json")
    with open(params_path, "w") as f:
        json.dump(params_dict, f, indent=2)
    logger.info(f"Identified params saved → {params_path}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Online GNN physical parameter fine-tuning")
    parser.add_argument("--config", type=str, default="configs/real.yaml")
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--enable_plasticity", action="store_true")
    parser.add_argument("--enable_breakage", action="store_true")
    parser.add_argument("--use_gnn_world_model", type=str, default="true")
    parser.add_argument("--gnn_online_finetune", type=str, default="true")
    parser.add_argument("--gnn_checkpoint_dir", type=str, default=None)
    parser.add_argument("--gnn_finetune_frames", type=int, default=None)
    parser.add_argument("--gnn_finetune_steps", type=int, default=None)
    parser.add_argument("--gnn_finetune_lr", type=float, default=None)
    args = parser.parse_args()

    cfg.load_from_yaml(args.config)
    for attr in ["gnn_checkpoint_dir", "gnn_finetune_frames", "gnn_finetune_steps", "gnn_finetune_lr"]:
        val = getattr(args, attr, None)
        if val is not None:
            setattr(cfg, attr, val)
    if args.enable_breakage:
        cfg.enable_breakage = True
    if args.enable_plasticity:
        cfg.enable_plasticity = True
    cfg.device = "cuda:0"

    run_gnn_online_finetune(args)
