"""Visualize GNN inference results as videos.

Supports three modes:
  1. Point-cloud video  (--mode pcd)   — lightweight 3D scatter render via matplotlib
  2. Side-by-side video (--mode compare) — GNN vs Warp inference point clouds
  3. Gaussian Splatting  (--mode gs)    — photorealistic render through the GS pipeline

Usage
-----
    # Quick point-cloud video for GNN inference
    python visualize_gnn_results.py --case_name demo_58 --mode pcd

    # Side-by-side GNN vs Warp (and optionally GNN-finetuned)
    python visualize_gnn_results.py --case_name demo_58 --mode compare

    # Side-by-side: all three (warp, gnn, gnn-finetuned)
    python visualize_gnn_results.py --case_name demo_58 --mode compare --include_finetuned

    # GS rendering (requires trained Gaussian model)
    python visualize_gnn_results.py --case_name demo_58 --mode gs

    # Specify which pkl to visualise
    python visualize_gnn_results.py --case_name demo_58 --mode pcd --pkl inference_gnn_finetuned.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_vertices(path: str) -> np.ndarray:
    """Load (n_frames, n_points, 3) from pkl."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return np.asarray(data, dtype=np.float32)


def _camera_params(base_path: str, case_name: str):
    """Return (c2ws, intrinsics, WH) from the dataset metadata."""
    with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        meta = json.load(f)
    return np.array(c2ws), np.array(meta["intrinsics"]), meta["WH"]


def _project(pts: np.ndarray, w2c: np.ndarray, K: np.ndarray):
    """Project (N,3) world points to (N,2) pixel coords via w2c + K."""
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    pts_h = np.concatenate([pts, ones], axis=1)  # (N,4)
    cam = (w2c @ pts_h.T).T  # (N,4) → keep first 3
    cam = cam[:, :3]
    # filter behind camera
    mask = cam[:, 2] > 0.01
    px = (K @ cam.T).T  # (N,3)
    with np.errstate(divide="ignore", invalid="ignore"):
        px = px[:, :2] / px[:, 2:3]
    return px, mask


class FFmpegWriter:
    """Write H.264 mp4 video via ffmpeg subprocess (universally playable)."""

    def __init__(self, path: str, width: int, height: int, fps: int = 30):
        # Ensure dimensions are even (H.264 requirement)
        self.width = width if width % 2 == 0 else width + 1
        self.height = height if height % 2 == 0 else height + 1
        self.path = path
        self.proc = subprocess.Popen(
            [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{self.width}x{self.height}",
                "-pix_fmt", "bgr24",
                "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "20",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame_bgr: np.ndarray):
        """Write one BGR frame (H, W, 3) uint8."""
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))
        self.proc.stdin.write(frame_bgr.tobytes())

    def release(self):
        self.proc.stdin.close()
        self.proc.wait()


# ---------------------------------------------------------------------------
# Mode 1: Point-cloud video
# ---------------------------------------------------------------------------

def render_pcd_video(
    vertices: np.ndarray,
    out_path: str,
    w2c: np.ndarray,
    K: np.ndarray,
    WH: list,
    fps: int = 30,
    point_size: float = 0.4,
    color: str = "dodgerblue",
    label: str = "",
):
    """Render a point-cloud video by projecting 3D points onto a 2D canvas."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    W, H = WH
    n_frames = vertices.shape[0]

    writer = FFmpegWriter(out_path, W, H, fps)

    for i in range(n_frames):
        fig, ax = plt.subplots(1, 1, figsize=(W / 100, H / 100), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("white")

        px, mask = _project(vertices[i], w2c, K)
        px = px[mask]
        ax.scatter(px[:, 0], px[:, 1], s=point_size, c=color, alpha=0.6, edgecolors="none")
        if label:
            ax.text(10, 25, f"{label}  frame {i}", fontsize=10, color="black",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(int(fig.get_figheight() * 100), int(fig.get_figwidth() * 100), 4)
        frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        # Resize to exact output dims in case dpi rounding causes mismatches
        if frame_bgr.shape[1] != W or frame_bgr.shape[0] != H:
            frame_bgr = cv2.resize(frame_bgr, (W, H))
        writer.write(frame_bgr)
        plt.close(fig)

    writer.release()
    print(f"  ✓ Saved → {out_path}  ({n_frames} frames, {W}×{H})")


# ---------------------------------------------------------------------------
# Mode 2: Side-by-side comparison video
# ---------------------------------------------------------------------------

def render_compare_video(
    results: dict[str, np.ndarray],
    out_path: str,
    w2c: np.ndarray,
    K: np.ndarray,
    WH: list,
    fps: int = 30,
    point_size: float = 0.4,
):
    """Side-by-side comparison of multiple rollouts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    W, H = WH
    n_panels = len(results)
    panel_w = W
    total_w = panel_w * n_panels
    n_frames = min(v.shape[0] for v in results.values())

    colors_map = {
        "Warp": "royalblue",
        "GNN": "crimson",
        "GNN-Finetuned": "forestgreen",
    }

    writer = FFmpegWriter(out_path, total_w, H, fps)

    for i in range(n_frames):
        fig, axes = plt.subplots(1, n_panels, figsize=(total_w / 100, H / 100), dpi=100)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0)
        if n_panels == 1:
            axes = [axes]

        for ax, (name, verts) in zip(axes, results.items()):
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_facecolor("white")

            px, mask = _project(verts[i], w2c, K)
            c = colors_map.get(name, "gray")
            ax.scatter(px[mask, 0], px[mask, 1], s=point_size, c=c, alpha=0.6, edgecolors="none")
            ax.text(10, 25, f"{name}  frame {i}", fontsize=10, color="black",
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(int(fig.get_figheight() * 100), int(fig.get_figwidth() * 100), 4)
        frame_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
        if frame_bgr.shape[1] != total_w or frame_bgr.shape[0] != H:
            frame_bgr = cv2.resize(frame_bgr, (total_w, H))
        writer.write(frame_bgr)
        plt.close(fig)

    writer.release()
    print(f"  ✓ Saved → {out_path}  ({n_frames} frames, {total_w}×{H}, {n_panels} panels)")


# ---------------------------------------------------------------------------
# Mode 3: Overlay on real camera images
# ---------------------------------------------------------------------------

def render_overlay_video(
    vertices: np.ndarray,
    out_path: str,
    image_dir: str,
    w2c: np.ndarray,
    K: np.ndarray,
    WH: list,
    fps: int = 30,
    cam_id: int = 0,
    point_size: int = 2,
    color: tuple = (0, 0, 255),
    label: str = "",
):
    """Overlay projected point cloud on real camera images."""

    W, H = WH
    n_frames = vertices.shape[0]

    writer = FFmpegWriter(out_path, W, H, fps)

    for i in range(n_frames):
        img_path = os.path.join(image_dir, str(cam_id), f"{i}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, str(cam_id), f"{i:05d}.png")
        if os.path.exists(img_path):
            frame = cv2.imread(img_path)
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H))
        else:
            frame = np.full((H, W, 3), 255, dtype=np.uint8)

        px, mask = _project(vertices[i], w2c, K)
        pts = px[mask].astype(np.int32)
        for p in pts:
            if 0 <= p[0] < W and 0 <= p[1] < H:
                cv2.circle(frame, (int(p[0]), int(p[1])), point_size, color, -1)

        if label:
            cv2.putText(frame, f"{label}  frame {i}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        writer.write(frame)

    writer.release()
    print(f"  ✓ Saved → {out_path}  ({n_frames} frames, {W}×{H})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize GNN inference results")
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--exp_dir", type=str, default="./experiments")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pcd", "compare", "overlay", "gs"],
        default="pcd",
        help="pcd: point-cloud video, compare: side-by-side, overlay: on real images, gs: Gaussian Splatting",
    )
    parser.add_argument("--pkl", type=str, default="inference_gnn.pkl",
                        help="Filename of the pkl to visualise (for pcd/overlay modes)")
    parser.add_argument("--include_finetuned", action="store_true",
                        help="Include GNN-finetuned in comparison")
    parser.add_argument("--cam_id", type=int, default=0, help="Camera index for projection")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--suffix", type=str, default="",
                        help="Experiment suffix (_ep, _brk, etc.)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory (default: experiments/<case>/videos/)")
    args = parser.parse_args()

    case = args.case_name
    exp_base = os.path.join(args.exp_dir, f"{case}{args.suffix}")
    assert os.path.isdir(exp_base), f"Experiment dir not found: {exp_base}"

    # Camera setup
    c2ws, intrinsics, WH = _camera_params(args.base_path, case)
    cam_id = args.cam_id
    w2c = np.linalg.inv(c2ws[cam_id])
    K = intrinsics[cam_id] if intrinsics.ndim == 3 else intrinsics

    out_dir = args.output_dir or os.path.join(exp_base, "videos")
    os.makedirs(out_dir, exist_ok=True)

    # ── Mode: pcd ──
    if args.mode == "pcd":
        pkl_path = os.path.join(exp_base, args.pkl)
        assert os.path.exists(pkl_path), f"Not found: {pkl_path}"
        verts = _load_vertices(pkl_path)
        label = Path(args.pkl).stem
        out_path = os.path.join(out_dir, f"{label}_cam{cam_id}.mp4")
        render_pcd_video(verts, out_path, w2c, K, WH, fps=args.fps, label=label)

    # ── Mode: overlay ──
    elif args.mode == "overlay":
        pkl_path = os.path.join(exp_base, args.pkl)
        assert os.path.exists(pkl_path), f"Not found: {pkl_path}"
        verts = _load_vertices(pkl_path)
        label = Path(args.pkl).stem
        image_dir = os.path.join(args.base_path, case, "color")
        out_path = os.path.join(out_dir, f"{label}_overlay_cam{cam_id}.mp4")
        render_overlay_video(verts, out_path, image_dir, w2c, K, WH,
                             fps=args.fps, cam_id=cam_id, label=label)

    # ── Mode: compare ──
    elif args.mode == "compare":
        results = {}

        warp_path = os.path.join(exp_base, "inference.pkl")
        if os.path.exists(warp_path):
            results["Warp"] = _load_vertices(warp_path)

        gnn_path = os.path.join(exp_base, "inference_gnn.pkl")
        if os.path.exists(gnn_path):
            results["GNN"] = _load_vertices(gnn_path)

        if args.include_finetuned:
            ft_path = os.path.join(exp_base, "inference_gnn_finetuned.pkl")
            if os.path.exists(ft_path):
                results["GNN-Finetuned"] = _load_vertices(ft_path)

        if not results:
            print("No inference pkl files found. Nothing to compare.")
            sys.exit(1)

        out_path = os.path.join(out_dir, f"compare_cam{cam_id}.mp4")
        render_compare_video(results, out_path, w2c, K, WH, fps=args.fps)

    # ── Mode: gs ──
    elif args.mode == "gs":
        # Symlink the GNN inference as inference.pkl and call the existing GS renderer
        pkl_path = os.path.join(exp_base, args.pkl)
        assert os.path.exists(pkl_path), f"Not found: {pkl_path}"
        inf_link = os.path.join(exp_base, "inference.pkl")
        inf_backup = os.path.join(exp_base, "inference_warp_backup.pkl")

        # Backup original inference.pkl if it exists
        backed_up = False
        if os.path.exists(inf_link) and not os.path.islink(inf_link):
            os.rename(inf_link, inf_backup)
            backed_up = True
            print(f"  Backed up original inference.pkl → {inf_backup}")

        try:
            # Symlink our GNN pkl as inference.pkl
            if os.path.exists(inf_link) or os.path.islink(inf_link):
                os.remove(inf_link)
            os.symlink(os.path.abspath(pkl_path), inf_link)
            print(f"  Linked {args.pkl} → inference.pkl")

            gs_model_dir = f"./gaussian_output/{case}"
            gs_dirs = [d for d in os.listdir(gs_model_dir) if os.path.isdir(os.path.join(gs_model_dir, d))]
            if not gs_dirs:
                print(f"No GS model found in {gs_model_dir}")
                sys.exit(1)
            model_path = os.path.join(gs_model_dir, gs_dirs[0])

            gs_output = os.path.join(out_dir, "gs_render")
            cmd = (
                f"python gs_render_dynamics.py "
                f"-m {model_path} "
                f"--name {case} "
                f"--output_dir {gs_output} "
                f"--skip_train"
            )
            print(f"  Running: {cmd}")
            os.system(cmd)
        finally:
            # Restore original inference.pkl
            if backed_up and os.path.exists(inf_backup):
                if os.path.islink(inf_link):
                    os.remove(inf_link)
                os.rename(inf_backup, inf_link)
                print(f"  Restored original inference.pkl")

    print("\nDone!")


if __name__ == "__main__":
    main()
