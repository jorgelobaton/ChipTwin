"""
GNN data generation script.

Uses the existing ChipTwin Warp simulator as a data factory to produce
graph-structured training data for the PhysicsGNN.

Usage
-----
    python gnn/generate_data.py --config configs/real.yaml \\
        --use_gnn_world_model true --gnn_generate_data true \\
        --gnn_num_trajectories 500 --gnn_output_dir data/gnn/ \\
        --base_path ./data/different_types --case_name demo_cable

The script:
    1. Loads the best ChipTwin checkpoint for the scene.
    2. For each trajectory, samples randomized physical parameters and
       controller motions, runs the Warp forward simulator, and saves
       per-frame graph snapshots.
    3. Splits into train/val (90/10) and stores split indices.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import pickle
import random
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# ── Ensure project root is importable ──────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from qqtt import InvPhyTrainerWarp
from qqtt.utils import cfg, logger

try:
    import warp as wp
except ImportError:
    wp = None


# ──────────────────────────────────────────────────────────────
# Random parameter sampling
# ──────────────────────────────────────────────────────────────

def _sample_params(n_springs: int, enable_breakage: bool,
                   ref_spring_Y_log: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Sample randomized per-spring physical parameters.

    If ``ref_spring_Y_log`` is provided (from the best ChipTwin checkpoint),
    sampling is centred around the real distribution with moderate
    perturbation.  Otherwise falls back to a broad uniform prior.

    Parameters
    ----------
    n_springs : int
    enable_breakage : bool
    ref_spring_Y_log : (n_springs,) or None – log-stiffness from checkpoint.

    Returns
    -------
    dict of tensors, all shape ``(n_springs,)``.
    """
    device = cfg.device

    if ref_spring_Y_log is not None:
        # Perturb around checkpoint values: ±20% in log space
        noise = torch.randn(n_springs, device=device) * 0.20 * ref_spring_Y_log.std().clamp(min=0.1)
        spring_Y_log = ref_spring_Y_log + noise
        spring_Y_log.clamp_(min=1.0)  # at least log(e)≈2.7
    else:
        spring_Y_min_log = math.log(cfg.spring_Y_min + 1.0)
        spring_Y_max_log = math.log(cfg.spring_Y_max)
        spring_Y_log = torch.empty(n_springs, device=device).uniform_(spring_Y_min_log, spring_Y_max_log)

    # yield_strain: perturb around config default (typically 0.05)
    ys_base = cfg.yield_strain
    ys = torch.empty(n_springs, device=device).uniform_(
        max(0.005, ys_base * 0.5), ys_base * 2.0
    )

    # hardening_factor: perturb around config default (typically 0.3)
    hf_base = cfg.hardening_factor
    hf = torch.empty(n_springs, device=device).uniform_(
        max(0.0, hf_base * 0.5), min(1.0, hf_base * 2.0)
    )

    params: Dict[str, torch.Tensor] = {
        "spring_Y_log": spring_Y_log,
        "yield_strain": ys,
        "hardening_factor": hf,
    }

    if enable_breakage:
        bs = torch.empty(n_springs, device=device).uniform_(0.05, 2.0)
        params["break_strain"] = bs

    return params


def _random_rotation_matrix(max_angle_rad: float, device: torch.device) -> torch.Tensor:
    """Sample a random 3D rotation matrix (axis-angle, uniform random axis)."""
    # Random axis (uniform on unit sphere)
    axis = torch.randn(3, device=device)
    axis = axis / axis.norm().clamp(min=1e-8)
    # Random angle in [-max_angle, +max_angle]
    angle = torch.empty(1, device=device).uniform_(-max_angle_rad, max_angle_rad).item()
    # Rodrigues' rotation formula
    K = torch.zeros(3, 3, device=device)
    K[0, 1] = -axis[2]; K[0, 2] = axis[1]
    K[1, 0] = axis[2];  K[1, 2] = -axis[0]
    K[2, 0] = -axis[1]; K[2, 1] = axis[0]
    R = torch.eye(3, device=device) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    return R


def _smooth_random_walk(
    n_frames: int, n_points: int, step_std: float, smoothing_window: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a temporally smooth random walk trajectory.

    Returns (n_frames, n_points, 3) — all points move coherently (rigid body).
    """
    # Generate per-frame increments for a single rigid motion
    increments = torch.randn(n_frames, 1, 3, device=device) * step_std
    increments[0] = 0.0
    # Cumulative sum → raw walk
    raw_walk = increments.cumsum(dim=0)  # (F, 1, 3)
    # Temporal smoothing via 1D convolution
    if smoothing_window > 1 and n_frames > smoothing_window:
        kernel = torch.ones(smoothing_window, device=device) / smoothing_window
        # Convolve each spatial dim independently
        walk_flat = raw_walk.squeeze(1).T  # (3, F)
        padded = torch.nn.functional.pad(walk_flat, (smoothing_window // 2, smoothing_window // 2), mode='replicate')
        smoothed = torch.nn.functional.conv1d(padded.unsqueeze(0), kernel.view(1, 1, -1).expand(3, 1, -1), groups=3)
        raw_walk = smoothed[0, :, :n_frames].T.unsqueeze(1)  # (F, 1, 3)
    return raw_walk.expand(n_frames, n_points, 3)


def _randomize_controller_trajectory(
    controller_points: torch.Tensor,
    frame_len: int,
    traj_idx: int = -1,
    num_trajectories: int = 100,
    object_centroid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Create a randomized controller trajectory with diverse grasp configurations.

    The randomization strategy produces *genuinely different* controller
    behaviours so the GNN sees diverse force patterns:

    - **~5 %**: Unperturbed original (baseline for the model).
    - **~30 %**: **Spatial offset** — translate ALL controller initial
      positions by a random 3D offset (5–40 mm), then replay the original
      motion from the new start.  This changes the controller↔object
      distance and thus the spring force pattern.
    - **~25 %**: **Rotated grasp** — rotate the entire controller cluster
      around the object centroid, keeping the motion pattern.  Different
      object nodes now feel the dominant forces.
    - **~20 %**: **Speed + direction variation** — scale speed ×[0.3, 2.0]
      and rotate motion ±30° around a random axis.
    - **~20 %**: **Synthetic trajectory** — replace the original motion
      with a smooth random walk whose per-frame magnitude matches the
      real statistics.

    Parameters
    ----------
    controller_points : (F, C, 3) – original controller trajectory.
    frame_len : int
    traj_idx : int
    num_trajectories : int
    object_centroid : (3,) or None – centroid of object points (frame 0).

    Returns
    -------
    (F, C, 3) randomized trajectory.
    """
    if controller_points is None:
        return None

    device = controller_points.device
    F, C, _ = controller_points.shape

    # Statistics of the original per-frame displacement
    orig_disp = controller_points[1:] - controller_points[:-1]  # (F-1, C, 3)
    disp_mag = orig_disp.norm(dim=-1)  # (F-1, C)
    mean_disp = disp_mag.mean().item()  # ~1.1 mm/frame for demo_58
    total_drift = (controller_points[-1] - controller_points[0]).norm(dim=-1).mean().item()

    # Controller cluster centroid (frame 0) and extent
    ctrl_centroid_0 = controller_points[0].mean(dim=0)  # (3,)
    ctrl_extent = (controller_points[0].max(dim=0).values -
                   controller_points[0].min(dim=0).values)  # (3,)
    ctrl_radius = ctrl_extent.norm().item() / 2.0  # half-diagonal

    # Object centroid for rotation centre
    pivot = object_centroid if object_centroid is not None else ctrl_centroid_0

    # ── Strategy selection ──
    frac = traj_idx / max(num_trajectories, 1)

    if frac < 0.05:
        # ── 5 %: unperturbed original ──
        return controller_points[:frame_len].clone()

    elif frac < 0.35:
        # ── 30 %: Spatial offset of controller start position ──
        # Shift the entire controller cluster by a random 3D vector.
        # Magnitude: 5–40 mm (0.005–0.040 in scene units).
        direction = torch.randn(3, device=device)
        direction = direction / direction.norm().clamp(min=1e-8)
        magnitude = torch.empty(1, device=device).uniform_(0.005, 0.040).item()
        offset = direction * magnitude  # (3,)

        # Apply offset to all frames (rigid shift)
        shifted = controller_points[:frame_len].clone()
        shifted = shifted + offset.unsqueeze(0).unsqueeze(0)  # broadcast (F, C, 3)
        return shifted

    elif frac < 0.60:
        # ── 25 %: Rotated grasp around object centroid ──
        # Rotate the controller cluster around the object so a different
        # "side" of the object experiences the controller forces.
        max_angle = math.radians(60)  # up to ±60°
        R = _random_rotation_matrix(max_angle, device)

        # Rotate positions relative to pivot
        pts = controller_points[:frame_len].clone()
        pts_centered = pts - pivot.unsqueeze(0).unsqueeze(0)
        pts_rotated = (pts_centered.reshape(-1, 3) @ R.T).reshape(pts.shape)
        return pts_rotated + pivot.unsqueeze(0).unsqueeze(0)

    elif frac < 0.80:
        # ── 20 %: Speed + direction variation ──
        speed_factor = torch.empty(1, device=device).uniform_(0.3, 2.0).item()
        motion = controller_points - controller_points[0:1]  # (F, C, 3)
        scaled_motion = motion * speed_factor

        # Random 3D rotation of motion (±30°)
        R = _random_rotation_matrix(math.radians(30), device)
        rotated_motion = (scaled_motion.reshape(-1, 3) @ R.T).reshape(F, C, 3)

        # Optionally add a spatial offset too
        offset = torch.randn(3, device=device) * 0.010  # ~10mm jitter
        rand_traj = controller_points[0:1] + offset + rotated_motion
        return rand_traj[:frame_len]

    else:
        # ── 20 %: Synthetic smooth random walk ──
        # Replace the original motion with a smooth random walk whose
        # per-frame step magnitude matches the real distribution.
        walk = _smooth_random_walk(
            n_frames=F, n_points=C,
            step_std=mean_disp * 1.5,  # slightly larger than real
            smoothing_window=7,
            device=device,
        )
        # Start from original frame-0 position + random small offset
        offset = torch.randn(3, device=device) * 0.015  # ~15mm
        rand_traj = controller_points[0:1] + offset + walk
        return rand_traj[:frame_len]


# ──────────────────────────────────────────────────────────────
# Single trajectory generation
# ──────────────────────────────────────────────────────────────

def _run_one_trajectory(
    trainer: InvPhyTrainerWarp,
    params: Dict[str, torch.Tensor],
    rand_controller: Optional[torch.Tensor],
    frame_len: int,
) -> List[Dict[str, Any]]:
    """Run one Warp forward trajectory and collect per-frame snapshots.

    Returns
    -------
    list of dicts, one per frame-pair ``(t, t+1)``, each containing:
        positions, velocities, next_positions, next_velocities,
        edge_indices, rest_lengths, physical params, delta_v (GT).
    """
    sim = trainer.simulator
    device = cfg.device

    # Apply sampled params
    sim.set_spring_Y(params["spring_Y_log"].clone())
    if "yield_strain" in params:
        sim.set_yield_strain(params["yield_strain"].clone())
    if "hardening_factor" in params:
        sim.set_hardening_factor(params["hardening_factor"].clone())
    if "break_strain" in params:
        sim.set_break_strain(params["break_strain"].mean().item())

    # Reset state
    if sim.enable_plasticity or sim.enable_breakage:
        sim.reset_rest_lengths()
    sim.set_init_state(sim.wp_init_vertices, sim.wp_init_velocities)

    # Optionally override controller points
    orig_ctrl = sim.controller_points
    if rand_controller is not None and sim.controller_points is not None:
        sim.controller_points = [
            rand_controller[i].to(device).contiguous() for i in range(rand_controller.shape[0])
        ]

    snapshots: List[Dict[str, Any]] = []
    dt_frame = 1.0 / cfg.FPS

    # ── Controller-aware data: concatenate controller nodes to object nodes ──
    N_obj = sim.num_object_points
    has_ctrl = (trainer.controller_points is not None
                and len(trainer.controller_points) > 0)
    ctrl_pts = sim.controller_points if has_ctrl else None  # per-frame (C,3)
    N_ctrl = ctrl_pts[0].shape[0] if has_ctrl else 0
    N_total = N_obj + N_ctrl

    # ALL springs (including controller-to-object)
    springs_torch = wp.to_torch(sim.wp_springs, requires_grad=False).clone()
    rest_lengths_torch = wp.to_torch(sim.wp_rest_lengths, requires_grad=False).clone()

    # Per-spring physical params – pad controller springs with defaults
    # (controller springs don't really have learned params, but we need
    #  values for the edge feature vector so the tensor sizes match)
    all_spring_Y = params["spring_Y_log"].clone()
    all_yield    = params["yield_strain"].clone()
    all_hard     = params["hardening_factor"].clone()
    all_break    = params["break_strain"].clone() if "break_strain" in params else None

    # is_controller flag: 0 for object, 1 for controller
    is_controller = torch.zeros(N_total, device=device)
    if N_ctrl > 0:
        is_controller[N_obj:] = 1.0

    # Initial object state
    obj_pos = wp.to_torch(sim.wp_states[0].wp_x, requires_grad=False).clone()
    obj_vel = wp.to_torch(sim.wp_states[0].wp_v, requires_grad=False).clone()

    # Build full state by concatenating controller nodes
    if has_ctrl:
        c_pos = ctrl_pts[0].to(device).float()
        c_vel = torch.zeros(N_ctrl, 3, device=device)
        prev_pos = torch.cat([obj_pos, c_pos], dim=0)
        prev_vel = torch.cat([obj_vel, c_vel], dim=0)
    else:
        prev_pos = obj_pos
        prev_vel = obj_vel

    for frame_idx in range(1, frame_len):
        if cfg.data_type == "real":
            sim.set_controller_target(frame_idx, pure_inference=True)
        if sim.object_collision_flag:
            sim.update_collision_graph()

        if cfg.use_graph:
            wp.capture_launch(sim.forward_graph)
        else:
            sim.step()

        cur_obj_pos = wp.to_torch(sim.wp_states[-1].wp_x, requires_grad=False).clone()
        cur_obj_vel = wp.to_torch(sim.wp_states[-1].wp_v, requires_grad=False).clone()

        # Build full current state with controller nodes
        if has_ctrl:
            c_pos_cur = ctrl_pts[frame_idx].to(device).float()
            c_pos_prev = ctrl_pts[frame_idx - 1].to(device).float() if frame_idx > 0 else c_pos_cur
            c_vel_cur = (c_pos_cur - c_pos_prev) / dt_frame
            cur_pos = torch.cat([cur_obj_pos, c_pos_cur], dim=0)
            cur_vel = torch.cat([cur_obj_vel, c_vel_cur], dim=0)
        else:
            cur_pos = cur_obj_pos
            cur_vel = cur_obj_vel

        # GT velocity delta
        delta_v = cur_vel - prev_vel

        snapshot = {
            "positions": prev_pos.cpu(),
            "velocities": prev_vel.cpu(),
            "next_positions": cur_pos.cpu(),
            "next_velocities": cur_vel.cpu(),
            "delta_v": delta_v.cpu(),
            "edge_indices": springs_torch.cpu(),
            "rest_lengths": rest_lengths_torch.cpu(),
            "spring_Y_log": all_spring_Y.cpu(),
            "yield_strain": all_yield.cpu(),
            "hardening_factor": all_hard.cpu(),
            "is_controller": is_controller.cpu(),
            "dt_frame": dt_frame,
        }
        if all_break is not None:
            snapshot["break_strain"] = all_break.cpu()

        snapshots.append(snapshot)

        # advance
        prev_pos = cur_pos
        prev_vel = cur_vel
        sim.set_init_state(sim.wp_states[-1].wp_x, sim.wp_states[-1].wp_v)

    # Restore original controller points
    sim.controller_points = orig_ctrl

    return snapshots


# ──────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────

def run_gnn_data_generation(args: argparse.Namespace) -> None:
    """Top-level data generation procedure.

    Parameters
    ----------
    args : Namespace with ``base_path``, ``case_name``, ``enable_plasticity``,
           ``enable_breakage``, and all GNN config fields loaded into ``cfg``.
    """
    base_path = args.base_path
    case_name = args.case_name

    # Build experiment dir (same convention as train_warp.py)
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

    logger.set_log_file(path=base_dir, name="gnn_data_gen_log")

    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        pure_inference_mode=True,
    )

    # Load best checkpoint
    ckpt_files = glob.glob(f"{base_dir}/train/best_*.pth")
    assert len(ckpt_files) > 0, f"No checkpoint found in {base_dir}/train/"
    ckpt_path = ckpt_files[0]
    checkpoint = torch.load(ckpt_path, map_location=cfg.device)
    spring_Y = checkpoint["spring_Y"] if "spring_Y" in checkpoint else None
    if spring_Y is not None:
        trainer.simulator.set_spring_Y(torch.log(spring_Y).detach().clone())

    n_springs = trainer.simulator.n_springs
    enable_breakage = cfg.enable_breakage
    num_traj = cfg.gnn_num_trajectories
    output_dir = cfg.gnn_output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Reference log-stiffness from checkpoint for centred sampling
    ref_spring_Y_log = torch.log(spring_Y).detach().to(cfg.device).float() if spring_Y is not None else None

    frame_len = trainer.dataset.frame_len
    controller_pts = None
    if trainer.controller_points is not None:
        # trainer.controller_points is already a list of (C, 3) torch tensors per frame
        controller_pts = torch.stack(
            [trainer.controller_points[i].float()
             for i in range(frame_len)], dim=0
        )  # (F, C, 3)

    # Object centroid for rotation-based augmentation
    obj_centroid = trainer.structure_points.float().mean(dim=0).to(cfg.device) if trainer.structure_points is not None else None

    logger.info(f"Generating {num_traj} trajectories → {output_dir}")
    for traj_idx in range(num_traj):
        params = _sample_params(n_springs, enable_breakage, ref_spring_Y_log=ref_spring_Y_log)
        rand_ctrl = _randomize_controller_trajectory(
            controller_pts, frame_len, traj_idx=traj_idx, num_trajectories=num_traj,
            object_centroid=obj_centroid,
        ) if controller_pts is not None else None
        snapshots = _run_one_trajectory(trainer, params, rand_ctrl, frame_len)
        traj_path = os.path.join(output_dir, f"traj_{traj_idx:05d}.pkl")
        with open(traj_path, "wb") as f:
            pickle.dump(snapshots, f)
        if (traj_idx + 1) % 50 == 0:
            logger.info(f"  [{traj_idx + 1}/{num_traj}] trajectories saved.")

    # Train/val split (90/10)
    indices = list(range(num_traj))
    random.shuffle(indices)
    split_point = int(0.9 * num_traj)
    split = {
        "train": indices[:split_point],
        "val": indices[split_point:],
    }
    split_path = os.path.join(output_dir, "split.json")
    with open(split_path, "w") as f:
        json.dump(split, f)

    logger.info(f"Data generation complete.  train={len(split['train'])}, val={len(split['val'])}")
    logger.info(f"Split saved to {split_path}")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GNN training data from Warp simulator")
    parser.add_argument("--config", type=str, default="configs/real.yaml")
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--enable_plasticity", action="store_true")
    parser.add_argument("--enable_breakage", action="store_true")
    parser.add_argument("--use_gnn_world_model", type=str, default="true")
    parser.add_argument("--gnn_generate_data", type=str, default="true")
    parser.add_argument("--gnn_num_trajectories", type=int, default=None)
    parser.add_argument("--gnn_output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg.load_from_yaml(args.config)
    if args.gnn_num_trajectories is not None:
        cfg.gnn_num_trajectories = args.gnn_num_trajectories
    if args.gnn_output_dir is not None:
        cfg.gnn_output_dir = args.gnn_output_dir

    run_gnn_data_generation(args)
