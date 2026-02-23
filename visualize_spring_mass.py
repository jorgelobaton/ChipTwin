"""Visualize learned spring-mass parameters for a trained case.

Produces (all written to  experiments/<case>[_ep][_brk]/params_vis/):
  - Histograms   : spring_Y, yield_strain, hardening_factor
  - Side-by-side comparison histogram  : spring_Y normal vs plasticity (if both exist)
  - Open3D LineSet .ply files coloured by value for each per-spring array
  - A printed summary table of every scalar/per-spring stat in the checkpoint

Usage
-----
    # Plasticity model only
    python visualize_spring_mass.py --case_name demo_64 --enable_plasticity

    # Compare normal and plasticity stiffness
    python visualize_spring_mass.py --case_name demo_64 --enable_plasticity --compare_normal

    # Show each LineSet in an interactive Open3D window right after saving
    python visualize_spring_mass.py --case_name demo_64 --enable_plasticity --show

    # Point at a specific checkpoint
    python visualize_spring_mass.py --case_name demo_64 --enable_plasticity \\
        --model_path experiments/demo_64_ep/train/best_100.pth
"""

import os
import glob
import pickle
import torch
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from qqtt.utils import logger, cfg


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lineset(points, springs, values, cmap_name="plasma", vmin=None, vmax=None):
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points)
    ls.lines  = o3d.utility.Vector2iVector(springs)
    vmin = float(values.min()) if vmin is None else vmin
    vmax = float(values.max()) if vmax is None else vmax
    normed = (values - vmin) / (vmax - vmin + 1e-12)
    colors = plt.get_cmap(cmap_name)(normed)[:, :3]
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def _save_histogram(values, title, xlabel, path, color="steelblue", bins=60):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(values, bins=bins, color=color, edgecolor="white", linewidth=0.4)
    mean, std = values.mean(), values.std()
    ax.axvline(mean, color="red",    linestyle="--", linewidth=1.2, label=f"mean={mean:.4g}")
    ax.axvline(mean - std, color="orange", linestyle=":", linewidth=1.0)
    ax.axvline(mean + std, color="orange", linestyle=":", linewidth=1.0, label=f"±1σ={std:.4g}")
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved histogram → {path}")


def _save_comparison_histogram(vals_a, label_a, vals_b, label_b,
                               title, xlabel, path):
    """Overlay two distributions on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(vals_a, bins=60, alpha=0.6, color="royalblue",  label=label_a, edgecolor="white", linewidth=0.3)
    ax.hist(vals_b, bins=60, alpha=0.6, color="darkorange", label=label_b, edgecolor="white", linewidth=0.3)
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"  Saved comparison histogram → {path}")


def _print_table(rows):
    if not rows:
        return
    width = max(len(r[0]) for r in rows) + 2
    sep   = "─" * (width + 32)
    print(sep)
    print(f"  {'Parameter':<{width}} Value")
    print(sep)
    for name, val in rows:
        if isinstance(val, float):
            print(f"  {name:<{width}} {val:.6g}")
        else:
            print(f"  {name:<{width}} {val}")
    print(sep)


def _build_topology_from_data(final_data_path, cma_params):
    """Rebuild object point cloud and spring topology from final_data.pkl."""
    with open(final_data_path, "rb") as f:
        final_data = pickle.load(f)

    pts_all = final_data.get("object_points") if isinstance(final_data, dict) else None
    if pts_all is None:
        return None, None

    pts0 = np.array(pts_all[0])

    obj_r  = float(cma_params.get("object_radius",        0.02)) if cma_params else 0.02
    obj_nn = int(  cma_params.get("object_max_neighbours", 30))  if cma_params else 30

    pcd  = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts0)
    tree = o3d.geometry.KDTreeFlann(pcd)

    flags   = np.zeros((len(pts0), len(pts0)), dtype=bool)
    springs = []
    for i in range(len(pts0)):
        [_, idx, _] = tree.search_hybrid_vector_3d(pts0[i], obj_r, obj_nn)
        for j in idx[1:]:
            if not flags[i, j] and np.linalg.norm(pts0[i] - pts0[j]) > 1e-4:
                flags[i, j] = flags[j, i] = True
                springs.append([i, j])

    return pts0, np.array(springs)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = ArgumentParser()
    parser.add_argument("--base_path",      type=str, default="./data/different_types")
    parser.add_argument("--case_name",      type=str, required=True)
    parser.add_argument("--model_path",     type=str, default=None,
                        help="Explicit checkpoint path; auto-detected if omitted.")
    parser.add_argument("--enable_plasticity", action="store_true")
    parser.add_argument("--enable_breakage",   action="store_true")
    parser.add_argument("--compare_normal",    action="store_true",
                        help="Also load the non-plasticity model and compare spring_Y distributions.")
    parser.add_argument("--show", action="store_true",
                        help="Open each LineSet in an Open3D viewer window right after saving.")
    args = parser.parse_args()

    case_name  = args.case_name
    base_path  = args.base_path

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    # Build experiment suffix & output dir
    suffix = ""
    if args.enable_plasticity:
        suffix += "_ep"
    if args.enable_breakage:
        suffix += "_brk"

    base_dir = f"experiments/{case_name}{suffix}"
    out_dir  = f"{base_dir}/params_vis"
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    # ── CMA params (for topology + scalar summary) ────────────────────────────
    opt_path   = f"experiments_optimization/{case_name}{suffix}/optimal_params.pkl"
    cma_params = None
    if os.path.exists(opt_path):
        with open(opt_path, "rb") as f:
            cma_params = pickle.load(f)
        logger.info(f"Loaded CMA params from {opt_path}")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    if args.model_path:
        ckpt_path = args.model_path
    else:
        candidates = sorted(glob.glob(f"{base_dir}/train/best_*.pth"))
        assert candidates, f"No best_*.pth in {base_dir}/train/ — run training first or pass --model_path."
        ckpt_path = candidates[0]

    logger.info(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    n_obj = ckpt.get("num_object_springs")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n╔══ Parameter Summary ══════════════════════════════════════════╗")

    if cma_params:
        print("\n  ── CMA-ES Stage ──")
        cma_rows = []
        for k in ["global_spring_Y", "object_radius", "object_max_neighbours",
                  "controller_radius", "controller_max_neighbours",
                  "collide_elas", "collide_fric", "collide_object_elas", "collide_object_fric",
                  "collision_dist", "drag_damping", "dashpot_damping",
                  "hardening_factor", "yield_strain", "break_strain"]:
            if k in cma_params:
                cma_rows.append((f"[CMA] {k}", float(cma_params[k])))
        _print_table(cma_rows)

    print("\n  ── Training Stage ──")
    train_rows = [("[Train] epoch", ckpt.get("epoch", "?")),
                  ("[Train] num_object_springs", n_obj)]
    for k in ["collide_elas", "collide_fric", "collide_object_elas", "collide_object_fric", "break_strain"]:
        if k in ckpt:
            v = ckpt[k]
            train_rows.append((f"[Train] {k}", float(v.item() if isinstance(v, torch.Tensor) else v)))
    for k in ["spring_Y", "yield_strain", "hardening_factor"]:
        if k in ckpt and isinstance(ckpt[k], torch.Tensor) and ckpt[k].numel() > 1:
            t  = ckpt[k][:n_obj] if n_obj else ckpt[k]
            train_rows += [
                (f"[Train] {k} mean", float(t.mean())),
                (f"[Train] {k} std",  float(t.std())),
                (f"[Train] {k} min",  float(t.min())),
                (f"[Train] {k} max",  float(t.max())),
            ]
    _print_table(train_rows)

    # ── Extract per-spring arrays (object springs only) ───────────────────────
    def _obj(key):
        t = ckpt.get(key)
        if t is None or not isinstance(t, torch.Tensor):
            return None
        return (t[:n_obj] if n_obj else t).numpy()

    spring_Y       = _obj("spring_Y")
    yield_strain   = _obj("yield_strain")
    hardening      = _obj("hardening_factor")

    # ── Histograms ────────────────────────────────────────────────────────────
    logger.info("\nSaving histograms …")

    if spring_Y is not None:
        _save_histogram(spring_Y, f"Spring Stiffness (Y) – {case_name}{suffix}",
                        "spring_Y  [N/m]", f"{out_dir}/hist_spring_Y.png", color="royalblue")

    if args.enable_plasticity:
        if yield_strain is not None:
            _save_histogram(yield_strain, f"Yield Strain – {case_name}{suffix}",
                            "yield_strain", f"{out_dir}/hist_yield_strain.png", color="darkorange")
        if hardening is not None:
            _save_histogram(hardening, f"Hardening Factor – {case_name}{suffix}",
                            "hardening_factor", f"{out_dir}/hist_hardening_factor.png", color="seagreen")

    # ── Optional: side-by-side comparison with non-plasticity model ───────────
    if args.compare_normal and spring_Y is not None:
        normal_candidates = sorted(glob.glob(f"experiments/{case_name}/train/best_*.pth"))
        if normal_candidates:
            ckpt_n = torch.load(normal_candidates[0], map_location="cpu")
            n_obj_n = ckpt_n.get("num_object_springs")
            t_n     = ckpt_n.get("spring_Y")
            if t_n is not None and isinstance(t_n, torch.Tensor):
                sy_n = (t_n[:n_obj_n] if n_obj_n else t_n).numpy()
                _save_comparison_histogram(
                    sy_n, "No plasticity",
                    spring_Y, f"Plasticity ({suffix.strip('_')})",
                    f"Spring Stiffness Comparison – {case_name}",
                    "spring_Y  [N/m]",
                    f"{out_dir}/hist_spring_Y_comparison.png",
                )
        else:
            logger.warning(f"  No normal model found at experiments/{case_name}/train/ – skipping comparison.")

    # ── 3-D LineSets ──────────────────────────────────────────────────────────
    final_data_path = f"{base_path}/{case_name}/final_data.pkl"
    if not os.path.exists(final_data_path):
        logger.warning(f"final_data.pkl not found at {final_data_path} – skipping 3-D renders.")
        return

    logger.info("\nBuilding spring topology …")
    pts0, springs = _build_topology_from_data(final_data_path, cma_params)
    if springs is None:
        logger.warning("  Could not extract topology – skipping 3-D renders.")
        return

    n_springs = len(springs)
    logger.info(f"  Topology: {len(pts0)} pts, {n_springs} springs")

    def _save_lineset(values, label, cmap_name, fname):
        if values is None:
            return
        vals = values[:n_springs]
        if len(vals) != n_springs:
            logger.warning(f"  {label}: length mismatch ({len(vals)} vs {n_springs}) – skipping.")
            return
        path = f"{out_dir}/{fname}"
        ls = _lineset(pts0, springs, vals, cmap_name)
        o3d.io.write_line_set(path, ls)
        logger.info(f"  Saved LineSet → {path}")
        if args.show:
            logger.info(f"  Showing '{label}' — close window to continue …")
            o3d.visualization.draw_geometries(
                [ls],
                window_name=f"{label} — {case_name}{suffix}",
                width=1280, height=720,
            )

    _save_lineset(spring_Y,     "spring_Y",       "plasma",  "lineset_spring_Y.ply")

    if args.enable_plasticity:
        _save_lineset(yield_strain, "yield_strain",   "magma",   "lineset_yield_strain.ply")
        _save_lineset(hardening,    "hardening_factor","viridis", "lineset_hardening_factor.ply")

        # Bonus: ratio  hardening / yield_strain  → shows work-hardening efficiency per spring
        if yield_strain is not None and hardening is not None:
            ratio = np.where(yield_strain > 1e-8, hardening / (yield_strain + 1e-8), 0.0)
            _save_lineset(ratio[:n_springs], "hardening/yield ratio", "coolwarm",
                          "lineset_hardening_yield_ratio.ply")
            _save_histogram(ratio[:n_springs],
                            f"Hardening/Yield Ratio – {case_name}{suffix}",
                            "hardening_factor / yield_strain",
                            f"{out_dir}/hist_hardening_yield_ratio.png",
                            color="mediumvioletred")

    logger.info(f"\n✓ All outputs written to: {out_dir}")
    print(f"\nOpen the .ply files in MeshLab or Cloud Compare to inspect spatial distributions.")


if __name__ == "__main__":
    main()
