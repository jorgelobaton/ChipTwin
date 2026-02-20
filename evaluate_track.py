import argparse
import pickle
import csv
import json
import numpy as np
from scipy.spatial import KDTree
import os
import matplotlib.pyplot as plt

base_path = "./data/different_types"
prediction_path = "experiments"
output_file = "results/final_track.csv"

if not os.path.exists("results"):
    os.makedirs("results")

parser = argparse.ArgumentParser(description="Evaluate tracking error")
parser.add_argument("--normal", action="store_true", help="Evaluate normal variant")
parser.add_argument("--plasticity", action="store_true", help="Evaluate plasticity variant")
parser.add_argument("--breakage", action="store_true", help="Evaluate breakage variant")
parser.add_argument("--case_name", type=str, default="", help="Evaluate only a specific case (by name) instead of all cases in data_config.csv")

def evaluate_prediction(start_frame, end_frame, vertices, gt_track_3d, idx, mask):
    track_errors = []
    for frame_idx in range(start_frame, end_frame):
        # Get the new mask and see
        new_mask = ~np.isnan(gt_track_3d[frame_idx][mask]).any(axis=1)
        gt_track_points = gt_track_3d[frame_idx][mask][new_mask]
        pred_x = vertices[frame_idx][idx][new_mask]
        if len(pred_x) == 0:
            track_error = 0
        else:
            track_error = np.mean(np.linalg.norm(pred_x - gt_track_points, axis=1))

        track_errors.append(track_error)
    return np.mean(track_errors)


def evaluate_case(case_name, exp_dir):
    """Evaluate a single experiment directory. Returns (train_error, test_error) or None if missing files."""
    inference_path = f"{exp_dir}/inference.pkl"
    gt_track_path = f"{base_path}/{case_name}/gt_track_3d.pkl"
    track_process_path = f"{base_path}/{case_name}/track_process_data.pkl"
    split_path = f"{base_path}/{case_name}/split.json"

    if not os.path.exists(inference_path):
        print(f"  Skipping {exp_dir}: inference.pkl not found")
        return None
    if not os.path.exists(split_path):
        print(f"  Skipping {exp_dir}: split.json not found")
        return None

    # Determine which GT tracking source to use
    if os.path.exists(gt_track_path):
        gt_source = "gt_track_3d"
    elif os.path.exists(track_process_path):
        gt_source = "track_process_data"
    else:
        print(f"  Skipping {exp_dir}: no tracking GT found (tried gt_track_3d.pkl, track_process_data.pkl)")
        return None

    with open(split_path, "r") as f:
        split = json.load(f)
    train_frame = split["train"][1]
    test_frame = split["test"][1]

    with open(inference_path, "rb") as f:
        vertices = pickle.load(f)

    if gt_source == "gt_track_3d":
        with open(gt_track_path, "rb") as f:
            gt_track_3d = pickle.load(f)
    else:
        # Build gt_track_3d from track_process_data.pkl
        # object_points: (frames, N, 3), object_visibilities: (frames, N)
        with open(track_process_path, "rb") as f:
            track_data = pickle.load(f)
        object_points = track_data["object_points"]       # (F, N, 3)
        object_vis = track_data["object_visibilities"]     # (F, N) bool
        # Set invisible points to NaN to match gt_track_3d format
        gt_track_3d = object_points.copy().astype(np.float64)
        gt_track_3d[~object_vis] = np.nan

    # Locate the index of corresponding point index in the vertices
    mask = ~np.isnan(gt_track_3d[0]).any(axis=1)
    kdtree = KDTree(vertices[0])
    _, idx = kdtree.query(gt_track_3d[0][mask])

    train_track_error = evaluate_prediction(1, train_frame, vertices, gt_track_3d, idx, mask)
    test_track_error = evaluate_prediction(train_frame, test_frame, vertices, gt_track_3d, idx, mask)

    return train_track_error, test_track_error


if __name__ == "__main__":
    args = parser.parse_args()

    if not (args.normal or args.plasticity or args.breakage):
        args.normal = True
        args.plasticity = True

    # Read case names
    if args.case_name:
        case_names = [args.case_name]
    else:
        case_names = []
        with open("data_config.csv", newline="", encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                if row:
                    case_names.append(row[0].strip())

    # Collect results for CSV and plotting
    results = []  # list of (case_name, variant_label, train_err, test_err)

    file = open(output_file, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(file)
    writer.writerow(["Case Name", "Variant", "Train Track Error", "Test Track Error"])

    for case_name in case_names:
        if not os.path.exists(f"{base_path}/{case_name}"):
            print(f"Data directory not found for {case_name}, skipping")
            continue

        print(f"Processing {case_name}")

        variants_to_run = []
        if args.normal:
            variants_to_run.append(("normal", ""))
        if args.plasticity:
            variants_to_run.append(("plasticity", "_ep"))
        if args.breakage:
            variants_to_run.append(("breakage", "_breakage"))

        for variant_name, suffix in variants_to_run:
            exp_dir = f"{prediction_path}/{case_name}{suffix}"
            result = evaluate_case(case_name, exp_dir)
            if result is not None:
                train_err, test_err = result
                writer.writerow([case_name, variant_name, train_err, test_err])
                results.append((case_name, variant_name, train_err, test_err))
                print(f"  {variant_name:<10} -> train: {train_err:.6f}, test: {test_err:.6f}")

    file.close()
    print(f"\nResults saved to {output_file}")

    # ── Plotting ──
    if not results:
        print("No results to plot.")
        exit()

    # Group results by case_name
    case_order = []
    seen = set()
    for case_name, _, _, _ in results:
        if case_name not in seen:
            case_order.append(case_name)
            seen.add(case_name)

    variants = []
    if args.normal: variants.append("normal")
    if args.plasticity: variants.append("plasticity")
    if args.breakage: variants.append("breakage")

    # Prepare data
    data = {"Train": {v: {} for v in variants}, "Test": {v: {} for v in variants}}
    for case_name, variant, train_err, test_err in results:
        data["Train"][variant][case_name] = train_err * 1000  # convert to mm for better visualization
        data["Test"][variant][case_name] = test_err * 1000  # convert to mm for better visualization

    x = np.arange(len(case_order))
    
    num_bars_per_case = len(variants) * 2
    total_width = 0.8
    bar_width = total_width / num_bars_per_case
    
    fig, ax = plt.subplots(figsize=(max(10, len(case_order) * num_bars_per_case * 0.6), 6))
    
    colors = {
        "normal": "#4C72B0",
        "plasticity": "#DD8452",
        "breakage": "#55A868"
    }
    
    for i, phase in enumerate(["Train", "Test"]):
        phase_offset = -total_width/4 if phase == "Train" else total_width/4
        
        for j, variant in enumerate(variants):
            variant_offset = (j - (len(variants) - 1) / 2) * bar_width
            pos = x + phase_offset + variant_offset
            
            vals = [data[phase][variant].get(c, 0) for c in case_order]
            has_val = [c in data[phase][variant] for c in case_order]
            
            label = f"{variant.capitalize()} ({phase})"
            hatch = "///" if phase == "Test" else ""
            alpha = 0.9 if phase == "Train" else 0.7
            
            bars = ax.bar(pos, vals, bar_width, label=label, color=colors[variant], alpha=alpha, hatch=hatch, edgecolor="white")
            
            for k, (v, h) in enumerate(zip(vals, has_val)):
                if h:
                    ax.text(pos[k], v + max(max(vals), 1e-5)*0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8, rotation=0)

    ax.set_xlabel("Case", fontweight="bold")
    ax.set_ylabel("Track Error (mm)", fontweight="bold")
    ax.set_title("Tracking Error", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(case_order, rotation=0, ha="center")
    
    for i in range(len(case_order) - 1):
        ax.axvline(x[i] + 0.5, color='gray', linestyle='--', alpha=0.3)
        
    ax.legend(bbox_to_anchor=(0, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("results/track_error_comparison.png", dpi=150, bbox_inches="tight")
    print("Plot saved to results/track_error_comparison.png")
    plt.show()
