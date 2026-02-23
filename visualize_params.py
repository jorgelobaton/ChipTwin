"""Visualize the learned parameters for a case after training / inference.

Produces:
  - A printed summary table of all scalar parameters (CMA stage + train stage).
  - Matplotlib histograms of the per-spring arrays (spring_Y, yield_strain,
    hardening_factor) saved to  experiments/<case>/params_vis/.
  - Open3D line-set renders of each per-spring array coloured by value, saved
    as .ply files in the same folder (open in MeshLab / Open3D viewer).

Usage
-----
    # Minimal – auto-finds best checkpoint
    python visualize_params.py --base_path ./data/different_types --case_name demo_64

    # With plasticity suffix
    python visualize_params.py --base_path ./data/different_types --case_name demo_64 --enable_plasticity

    # Point at a specific checkpoint
    python visualize_params.py --base_path ./data/different_types --case_name demo_64 \\
        --model_path experiments/demo_64_ep/train/best_42.pth --enable_plasticity
"""

import os
import glob
import json
import pickle
import argparse

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless – no display required
import matplotlib.pyplot as plt
import open3d as o3d

from qqtt.utils import cfg, logger


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lineset(points, springs, values, cmap_name="plasma", vmin=None, vmax=None):
    """Build a coloured Open3D LineSet from per-spring scalar values."""
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines  = o3d.utility.Vector2iVector(springs)

    vmin = values.min() if vmin is None else vmin
    vmax = values.max() if vmax is None else vmax
    normed = (values - vmin) / (vmax - vmin + 1e-12)

    cmap   = plt.get_cmap(cmap_name)
    colors = cmap(normed)[:, :3]
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def _save_histogram(values, title, xlabel, out_path, color="steelblue"):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=60, color=color, edgecolor="white", linewidth=0.4)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved histogram → {out_path}")


def _print_table(rows):
    """rows: list of (name, value) tuples."""
    width = max(len(r[0]) for r in rows) + 2
    sep   = "─" * (width + 30)
    print(sep)
    print(f"{'Parameter':<{width}}  Value")
    print(sep)
    for name, val in rows:
        if isinstance(val, float):
            print(f"  {name:<{width-2}}  {val:.6g}")
        else:
            print(f"  {name:<{width-2}}  {val}")
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize learned parameters for a case."
    )
    parser.add_argument("--base_path",  type=str, required=True)
    parser.add_argument("--case_name",  type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to a specific .pth checkpoint. "
                             "If omitted, the best_*.pth in the experiment dir is used.")
    parser.add_argument("--enable_plasticity", action="store_true")
    parser.add_argument("--enable_breakage",   action="store_true")
    args = parser.parse_args()

    base_path  = args.base_path
    case_name  = args.case_name

    # ── Config ────────────────────────────────────────────────────────────────
    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    if args.enable_plasticity:
        cfg.enable_plasticity = True
    if args.enable_breakage:
        cfg.enable_breakage = True

    suffix = ""
    if args.enable_plasticity:
        suffix += "_ep"
    if args.enable_breakage:
        suffix += "_brk"

    base_dir  = f"experiments/{case_name}{suffix}"
    out_dir   = f"{base_dir}/params_vis"
    os.makedirs(out_dir, exist_ok=True)

    # ── CMA (stage-1) params ──────────────────────────────────────────────────
    opt_path = f"experiments_optimization/{case_name}{suffix}/optimal_params.pkl"
    cma_params = None
    if os.path.exists(opt_path):
        with open(opt_path, "rb") as f:
            cma_params = pickle.load(f)
        logger.info(f"Loaded CMA params from {opt_path}")
    else:
        logger.warning(f"CMA optimal_params.pkl not found at {opt_path} – skipping.")

    # ── Train (stage-2) checkpoint ────────────────────────────────────────────
    if args.model_path:
        ckpt_path = args.model_path
    else:
        candidates = sorted(glob.glob(f"{base_dir}/train/best_*.pth"))
        assert candidates, (
            f"No best_*.pth found in {base_dir}/train/. "
            "Run training first or pass --model_path."
        )
        ckpt_path = candidates[0]

    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # ── Print scalar summary ──────────────────────────────────────────────────
    rows_cma = []
    if cma_params is not None:
        scalar_cma_keys = [
            "global_spring_Y", "object_radius", "object_max_neighbours",
            "controller_radius", "controller_max_neighbours",
            "collide_elas", "collide_fric",
            "collide_object_elas", "collide_object_fric",
            "collision_dist", "drag_damping", "dashpot_damping",
            "hardening_factor", "yield_strain", "break_strain",
        ]
        for k in scalar_cma_keys:
            if k in cma_params:
                rows_cma.append((f"[CMA] {k}", float(cma_params[k])))

    rows_ckpt = []
    scalar_ckpt_keys = [
        "collide_elas", "collide_fric",
        "collide_object_elas", "collide_object_fric",
        "break_strain",
    ]
    for k in scalar_ckpt_keys:
        if k in ckpt:
            v = ckpt[k]
            rows_ckpt.append((f"[Train] {k}", float(v.item() if isinstance(v, torch.Tensor) else v)))

    rows_ckpt.append(("[Train] epoch", ckpt.get("epoch", "?")))
    rows_ckpt.append(("[Train] num_object_springs", ckpt.get("num_object_springs", "?")))

    # Per-spring stats
    per_spring_keys = ["spring_Y", "yield_strain", "hardening_factor"]
    for k in per_spring_keys:
        if k in ckpt:
            t = ckpt[k]
            if isinstance(t, torch.Tensor) and t.numel() > 1:
                rows_ckpt.append((f"[Train] {k} mean", float(t.mean())))
                rows_ckpt.append((f"[Train] {k} std",  float(t.std())))
                rows_ckpt.append((f"[Train] {k} min",  float(t.min())))
                rows_ckpt.append((f"[Train] {k} max",  float(t.max())))

    print("\n╔══ Parameter Summary ════════════════════════════════════╗")
    if rows_cma:
        print("\n  ── CMA-ES Stage ──")
        _print_table(rows_cma)
    print("\n  ── Training Stage ──")
    _print_table(rows_ckpt)

    # ── Histograms ────────────────────────────────────────────────────────────
    logger.info("\nSaving histograms …")
    spring_Y = ckpt.get("spring_Y")
    if spring_Y is not None and isinstance(spring_Y, torch.Tensor):
        num_obj_springs = ckpt.get("num_object_springs")
        sy_obj = spring_Y[:num_obj_springs].numpy() if num_obj_springs else spring_Y.numpy()
        _save_histogram(
            sy_obj, f"spring_Y (object springs) – {case_name}",
            "spring_Y  [N/m]",
            f"{out_dir}/hist_spring_Y.png",
            color="royalblue",
        )

    if args.enable_plasticity:
        ys = ckpt.get("yield_strain")
        hf = ckpt.get("hardening_factor")
        if ys is not None and isinstance(ys, torch.Tensor):
            _save_histogram(
                ys.numpy(), f"yield_strain – {case_name}",
                "yield_strain",
                f"{out_dir}/hist_yield_strain.png",
                color="darkorange",
            )
        if hf is not None and isinstance(hf, torch.Tensor):
            _save_histogram(
                hf.numpy(), f"hardening_factor – {case_name}",
                "hardening_factor",
                f"{out_dir}/hist_hardening_factor.png",
                color="seagreen",
            )

    # ── Open3D LineSet renders ────────────────────────────────────────────────
    # We need the point / spring topology. Load it from CMA optimal_params only
    # (no GPU simulator needed here – just re-build the KD-tree topology).

    final_data_path = f"{base_path}/{case_name}/final_data.pkl"
    if not os.path.exists(final_data_path):
        logger.warning(f"final_data.pkl not found at {final_data_path} – skipping 3-D renders.")
    else:
        logger.info("\nBuilding spring topology for 3-D renders …")
        import open3d as o3d

        with open(final_data_path, "rb") as f:
            final_data = pickle.load(f)

        # Extract the first-frame object point cloud
        # final_data is expected to have an "object_points" key with shape (T, N, 3)
        if isinstance(final_data, dict):
            pts_all = final_data.get("object_points", None)
        else:
            # May be a list/tuple depending on data format
            pts_all = None

        if pts_all is not None:
            pts0 = np.array(pts_all[0]) if not isinstance(pts_all[0], np.ndarray) else pts_all[0]

            # Replicate spring-building logic from trainer using the CMA topology params
            if cma_params is not None:
                obj_r   = float(cma_params.get("object_radius",        0.02))
                obj_nn  = int(cma_params.get("object_max_neighbours",  30))
            else:
                obj_r, obj_nn = 0.02, 30

            pcd  = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts0)
            tree = o3d.geometry.KDTreeFlann(pcd)

            flags   = np.zeros((len(pts0), len(pts0)), dtype=bool)
            springs = []
            for i in range(len(pts0)):
                [_, idx, _] = tree.search_hybrid_vector_3d(pts0[i], obj_r, obj_nn)
                for j in idx[1:]:
                    rl = np.linalg.norm(pts0[i] - pts0[j])
                    if not flags[i, j] and rl > 1e-4:
                        flags[i, j] = flags[j, i] = True
                        springs.append([i, j])

            springs = np.array(springs)
            n_springs = len(springs)
            logger.info(f"  Topology: {len(pts0)} points, {n_springs} springs")

            def _try_save_lineset(values_tensor, label, cmap_name, fname):
                if values_tensor is None:
                    return
                vals = values_tensor.numpy() if isinstance(values_tensor, torch.Tensor) else np.array(values_tensor)
                # Trim or pad to match topology spring count
                vals = vals[:n_springs]
                if len(vals) != n_springs:
                    logger.warning(f"  {label}: length mismatch ({len(vals)} vs {n_springs}) – skipping.")
                    return
                ls = _lineset(pts0, springs, vals, cmap_name)
                path = f"{out_dir}/{fname}"
                o3d.io.write_line_set(path, ls)
                logger.info(f"  Saved LineSet → {path}")

            _try_save_lineset(ckpt.get("spring_Y"), "spring_Y", "plasma",  f"lineset_spring_Y.ply")
            if args.enable_plasticity:
                _try_save_lineset(ckpt.get("yield_strain"),     "yield_strain",     "magma",  "lineset_yield_strain.ply")
                _try_save_lineset(ckpt.get("hardening_factor"), "hardening_factor", "viridis","lineset_hardening_factor.ply")
        else:
            logger.warning("  Could not extract object_points from final_data.pkl – skipping 3-D renders.")

    logger.info(f"\nAll outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
