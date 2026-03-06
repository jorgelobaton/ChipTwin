"""
GNN inference script – fast forward rollout.

When ``use_gnn_world_model=true`` and ``gnn_train=false`` and
``gnn_generate_data=false`` and ``gnn_online_finetune=false``, this
module loads the trained GNN and runs a full forward rollout over all
frames, replacing the Warp forward pass.

Output is saved to ``inference_gnn.pkl`` in the same format as the
existing ``inference.pkl``.

Usage
-----
    python gnn/inference.py --config configs/real.yaml \\
        --base_path ./data/different_types --case_name demo_cable \\
        --use_gnn_world_model true
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
import sys
from typing import Optional

import numpy as np
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg, logger

try:
    import warp as wp
except ImportError:
    wp = None

from gnn.model import PhysicsGNN
from gnn.utils import build_graph_from_sim_state, load_gnn_checkpoint


def run_gnn_inference(args: argparse.Namespace) -> None:
    """Run a full GNN forward rollout and save ``inference_gnn.pkl``.

    Physical parameters are loaded from the best ChipTwin checkpoint and
    mapped to per-edge features consumed by the GNN.

    Parameters
    ----------
    args : Namespace with ``base_path``, ``case_name``, and related flags.
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
    assert os.path.exists(optimal_path), f"Optimal params not found: {optimal_path}"
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

    logger.set_log_file(path=base_dir, name="gnn_inference_log")

    # Build trainer (for init graph structure and loading checkpoint)
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )
    sim = trainer.simulator

    # Load best checkpoint physical params
    ckpt_files = glob.glob(f"{base_dir}/train/best_*.pth")
    assert len(ckpt_files) > 0, f"No checkpoint found in {base_dir}/train/"
    checkpoint = torch.load(ckpt_files[0], map_location=device)

    spring_Y = checkpoint["spring_Y"]
    spring_Y_log = torch.log(spring_Y).to(device).float()

    ys_ckpt = checkpoint.get("yield_strain")
    if ys_ckpt is not None:
        if not isinstance(ys_ckpt, torch.Tensor):
            ys_ckpt = torch.full((sim.n_springs,), float(ys_ckpt), device=device)
        ys = ys_ckpt.to(device).float()
    else:
        ys = torch.full((sim.n_springs,), cfg.yield_strain, device=device)

    hf_ckpt = checkpoint.get("hardening_factor")
    if hf_ckpt is not None:
        if not isinstance(hf_ckpt, torch.Tensor):
            hf_ckpt = torch.full((sim.n_springs,), float(hf_ckpt), device=device)
        hf = hf_ckpt.to(device).float()
    else:
        hf = torch.full((sim.n_springs,), cfg.hardening_factor, device=device)

    bs = None
    if enable_breakage:
        bs_ckpt = checkpoint.get("break_strain")
        if bs_ckpt is not None:
            if not isinstance(bs_ckpt, torch.Tensor):
                bs_ckpt = torch.tensor(bs_ckpt, device=device)
            bs = bs_ckpt.expand(sim.n_springs).to(device).float()
        else:
            bs = torch.full((sim.n_springs,), cfg.break_strain, device=device)

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

    # Controller mask (bool) for position override
    ctrl_mask = is_ctrl.bool()  # (N_total,)

    # Load GNN
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

    # Initial state from simulator
    # Must call set_init_state to populate wp_states[0] with actual positions
    if sim.enable_plasticity or sim.enable_breakage:
        sim.reset_rest_lengths()
    sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)

    obj_pos = wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False).float().to(device)  # (N_obj, 3)
    obj_vel = wp.to_torch(sim.wp_states[0].wp_v, requires_grad=False).float().to(device)  # (N_obj, 3)

    # Build full initial state with controller nodes
    if has_controller:
        ctrl_pos_0 = ctrl_traj[0]  # (N_ctrl, 3)
        ctrl_vel_0 = torch.zeros(N_ctrl, 3, device=device)
        pos = torch.cat([obj_pos, ctrl_pos_0], dim=0)  # (N_total, 3)
        vel = torch.cat([obj_vel, ctrl_vel_0], dim=0)  # (N_total, 3)
    else:
        pos = obj_pos
        vel = obj_vel

    dt_frame = 1.0 / cfg.FPS
    frame_len = trainer.dataset.frame_len

    # Save only object node positions
    vertices = [pos[:N_obj].cpu()]

    logger.info(f"Running GNN rollout for {frame_len} frames "
                f"(N_obj={N_obj}, N_ctrl={N_ctrl}, N_total={N_total}) …")
    with torch.no_grad():
        for frame_idx in range(1, frame_len):
            # Controller target for next frame
            ctrl_target = None
            if has_controller and frame_idx < ctrl_traj.shape[0]:
                ctrl_target = ctrl_traj[frame_idx]  # (N_ctrl, 3)

            pos, vel = model.rollout_step(
                pos, vel, edge_indices_torch, rest_lengths_torch,
                spring_Y_log, ys, hf, bs, is_ctrl,
                dt_frame, enable_breakage,
                controller_mask=ctrl_mask if has_controller else None,
                controller_target_pos=ctrl_target,
            )
            vertices.append(pos[:N_obj].cpu())

    vertices = torch.stack(vertices, dim=0).numpy()  # (T, N_obj, 3)

    save_path = os.path.join(base_dir, "inference_gnn.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(vertices, f)
    logger.info(f"GNN inference saved → {save_path}  shape={vertices.shape}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN forward rollout inference")
    parser.add_argument("--config", type=str, default="configs/real.yaml")
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--enable_plasticity", action="store_true")
    parser.add_argument("--enable_breakage", action="store_true")
    parser.add_argument("--use_gnn_world_model", type=str, default="true")
    parser.add_argument("--gnn_checkpoint_dir", type=str, default=None)
    args = parser.parse_args()

    cfg.load_from_yaml(args.config)
    if args.gnn_checkpoint_dir is not None:
        cfg.gnn_checkpoint_dir = args.gnn_checkpoint_dir
    cfg.device = "cuda:0"

    run_gnn_inference(args)
