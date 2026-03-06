"""Visualize GNN training trajectories as videos.

Renders one or more training trajectories from data/gnn/ as point-cloud
videos, showing object nodes in blue and controller nodes in red.
Optionally shows multiple trajectories side-by-side so you can see the
effect of randomized controller motions and physical parameters.

Usage
-----
    # Single trajectory (default: traj 0)
    python visualize_gnn_training_data.py --case_name demo_58

    # Specific trajectory
    python visualize_gnn_training_data.py --case_name demo_58 --traj_ids 0 5 10

    # Grid of N trajectories (auto-selected)
    python visualize_gnn_training_data.py --case_name demo_58 --grid 4

    # Use a specific camera view
    python visualize_gnn_training_data.py --case_name demo_58 --cam_id 0
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# FFmpegWriter (same as visualize_gnn_results.py)
# ---------------------------------------------------------------------------

class FFmpegWriter:
    """Write H.264 mp4 video via ffmpeg subprocess (universally playable)."""

    def __init__(self, path: str, width: int, height: int, fps: int = 30):
        self.width = width if width % 2 == 0 else width + 1
        self.height = height if height % 2 == 0 else height + 1
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", str(fps),
                "-i", "pipe:",
                "-c:v", "libx264", "-preset", "fast",
                "-crf", "20", "-pix_fmt", "yuv420p",
                self.path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self.proc.stdin.write(frame.tobytes())

    def release(self):
        self.proc.stdin.close()
        self.proc.wait()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_trajectory_positions(path: str):
    """Load a trajectory pkl and return (positions, is_controller) arrays.

    Returns
    -------
    positions : (T+1, N, 3)  – frame 0 is the initial positions, rest from next_positions
    is_controller : (N,)     – bool mask for controller nodes
    """
    with open(path, "rb") as f:
        snapshots = pickle.load(f)

    frames = [snapshots[0]["positions"].numpy()]
    for snap in snapshots:
        frames.append(snap["next_positions"].numpy())

    positions = np.stack(frames, axis=0)  # (T+1, N, 3)
    is_ctrl = snapshots[0]["is_controller"].numpy().astype(bool)  # (N,)
    return positions, is_ctrl


def camera_params(base_path: str, case_name: str):
    """Return (c2ws, intrinsics, W, H)."""
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        meta = json.load(f)
    wh = meta["WH"]
    # WH may be [W,H] or [[W,H],[W,H],...]
    if isinstance(wh[0], (list, tuple)):
        W, H = wh[0]
    else:
        W, H = wh
    return np.array(c2ws), np.array(meta["intrinsics"]), int(W), int(H)


def project(pts: np.ndarray, w2c: np.ndarray, K: np.ndarray):
    """Project (N,3) → (N,2) pixel coords."""
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    pts_h = np.concatenate([pts, ones], axis=1)
    cam = (w2c @ pts_h.T).T[:, :3]
    mask = cam[:, 2] > 0.01
    px = (K @ cam.T).T
    with np.errstate(divide="ignore", invalid="ignore"):
        px = px[:, :2] / px[:, 2:3]
    return px, mask


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_frame(
    positions: np.ndarray,
    is_ctrl: np.ndarray,
    w2c: np.ndarray,
    K: np.ndarray,
    W: int,
    H: int,
    title: str = "",
    point_size: int = 1,
) -> np.ndarray:
    """Render a single frame: blue=object, red=controller."""
    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    obj_pts = positions[~is_ctrl]
    ctrl_pts = positions[is_ctrl]

    # Object nodes (blue)
    px_obj, mask_obj = project(obj_pts, w2c, K)
    for i in np.where(mask_obj)[0]:
        x, y = int(px_obj[i, 0]), int(px_obj[i, 1])
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(canvas, (x, y), point_size, (255, 180, 50), -1)  # blue-ish

    # Controller nodes (red, larger)
    px_ctrl, mask_ctrl = project(ctrl_pts, w2c, K)
    for i in np.where(mask_ctrl)[0]:
        x, y = int(px_ctrl[i, 0]), int(px_ctrl[i, 1])
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(canvas, (x, y), point_size + 2, (0, 0, 255), -1)  # red

    if title:
        cv2.putText(canvas, title, (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def render_single_trajectory(
    traj_path: str,
    base_path: str,
    case_name: str,
    cam_id: int,
    output_path: str,
    fps: int = 30,
):
    """Render a single trajectory to a video file."""
    positions, is_ctrl = load_trajectory_positions(traj_path)
    c2ws, intrinsics, W, H = camera_params(base_path, case_name)
    K = intrinsics[cam_id]
    w2c = np.linalg.inv(c2ws[cam_id])
    n_frames = positions.shape[0]

    traj_name = Path(traj_path).stem

    writer = FFmpegWriter(output_path, W, H, fps)
    for t in range(n_frames):
        frame = render_frame(
            positions[t], is_ctrl, w2c, K, W, H,
            title=f"{traj_name}  frame {t}/{n_frames-1}",
        )
        writer.write(frame)
    writer.release()
    print(f"  ✓ Saved → {output_path}  ({n_frames} frames)")


def render_grid(
    traj_paths: List[str],
    base_path: str,
    case_name: str,
    cam_id: int,
    output_path: str,
    fps: int = 30,
    cols: int = 2,
):
    """Render multiple trajectories in a grid layout."""
    n = len(traj_paths)
    rows = (n + cols - 1) // cols

    # Load all trajectories
    all_positions = []
    all_ctrl = []
    for p in traj_paths:
        pos, ctrl = load_trajectory_positions(p)
        all_positions.append(pos)
        all_ctrl.append(ctrl)

    c2ws, intrinsics, W, H = camera_params(base_path, case_name)
    K = intrinsics[cam_id]
    w2c = np.linalg.inv(c2ws[cam_id])

    # Tile dimensions
    tile_w, tile_h = W, H
    grid_w = cols * tile_w
    grid_h = rows * tile_h
    n_frames = min(pos.shape[0] for pos in all_positions)

    writer = FFmpegWriter(output_path, grid_w, grid_h, fps)
    for t in range(n_frames):
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for idx in range(n):
            r, c = divmod(idx, cols)
            traj_name = Path(traj_paths[idx]).stem
            tile = render_frame(
                all_positions[idx][t], all_ctrl[idx], w2c, K, tile_w, tile_h,
                title=f"{traj_name}  f{t}",
            )
            y0 = r * tile_h
            x0 = c * tile_w
            grid[y0:y0+tile_h, x0:x0+tile_w] = tile
        writer.write(grid)
    writer.release()
    print(f"  ✓ Saved → {output_path}  ({n_frames} frames, {cols}×{rows} grid)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize GNN training trajectories"
    )
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--case_name", type=str, default="demo_58")
    parser.add_argument("--data_dir", type=str, default="data/gnn",
                        help="Directory containing traj_*.pkl files")
    parser.add_argument("--traj_ids", type=int, nargs="+", default=None,
                        help="Specific trajectory indices to render (e.g. 0 5 10)")
    parser.add_argument("--grid", type=int, default=None,
                        help="Render N trajectories in a grid (auto-selects evenly spaced)")
    parser.add_argument("--cam_id", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: experiments/<case>/videos/)")
    args = parser.parse_args()

    data_dir = args.data_dir
    case_name = args.case_name
    output_dir = args.output_dir or f"experiments/{case_name}/videos"
    os.makedirs(output_dir, exist_ok=True)

    # Discover available trajectories
    import glob
    traj_files = sorted(glob.glob(os.path.join(data_dir, "traj_*.pkl")))
    if not traj_files:
        print(f"No trajectories found in {data_dir}")
        sys.exit(1)
    print(f"Found {len(traj_files)} trajectories in {data_dir}")

    if args.grid is not None:
        # Grid mode: evenly-spaced selection
        n = min(args.grid, len(traj_files))
        step = max(1, len(traj_files) // n)
        selected = [traj_files[i * step] for i in range(n)]
        cols = 2 if n <= 4 else 3
        out_path = os.path.join(output_dir, f"training_grid_{n}_cam{args.cam_id}.mp4")
        print(f"Rendering {n} trajectories in a {cols}-column grid …")
        render_grid(selected, args.base_path, case_name, args.cam_id, out_path, args.fps, cols)

    elif args.traj_ids is not None:
        # Specific trajectories
        if len(args.traj_ids) == 1:
            idx = args.traj_ids[0]
            path = os.path.join(data_dir, f"traj_{idx:05d}.pkl")
            assert os.path.exists(path), f"Not found: {path}"
            out_path = os.path.join(output_dir, f"training_traj{idx:05d}_cam{args.cam_id}.mp4")
            print(f"Rendering trajectory {idx} …")
            render_single_trajectory(path, args.base_path, case_name, args.cam_id, out_path, args.fps)
        else:
            paths = []
            for idx in args.traj_ids:
                p = os.path.join(data_dir, f"traj_{idx:05d}.pkl")
                assert os.path.exists(p), f"Not found: {p}"
                paths.append(p)
            cols = min(len(paths), 3)
            out_path = os.path.join(
                output_dir,
                f"training_traj{'_'.join(str(i) for i in args.traj_ids)}_cam{args.cam_id}.mp4",
            )
            print(f"Rendering {len(paths)} trajectories in grid …")
            render_grid(paths, args.base_path, case_name, args.cam_id, out_path, args.fps, cols)
    else:
        # Default: trajectory 0
        path = traj_files[0]
        out_path = os.path.join(output_dir, f"training_traj00000_cam{args.cam_id}.mp4")
        print(f"Rendering trajectory 0 …")
        render_single_trajectory(path, args.base_path, case_name, args.cam_id, out_path, args.fps)

    print("\nDone!")


if __name__ == "__main__":
    main()
