import argparse
import glob
import pickle
import json
import torch
import csv
import numpy as np
import os
import matplotlib.pyplot as plt
from pytorch3d.loss import chamfer_distance

prediction_dir = "./experiments"
base_path = "./data/different_types"
output_file = "results/final_results.csv"

if not os.path.exists("results"):
    os.makedirs("results")

parser = argparse.ArgumentParser(description="Evaluate chamfer error")
parser.add_argument("--normal", action="store_true", help="Evaluate normal variant")
parser.add_argument("--plasticity", action="store_true", help="Evaluate plasticity variant")
parser.add_argument("--breakage", action="store_true", help="Evaluate breakage variant")
parser.add_argument("--case_name", type=str, default="", help="Evaluate only a specific case (by name) instead of all cases in data_config.csv")

def evaluate_prediction(
    start_frame,
    end_frame,
    vertices,
    object_points,
    object_visibilities,
    object_motions_valid,
    num_original_points,
    num_surface_points,
):
    chamfer_errors = []

    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    if not isinstance(object_points, torch.Tensor):
        object_points = torch.tensor(object_points, dtype=torch.float32)
    if not isinstance(object_visibilities, torch.Tensor):
        object_visibilities = torch.tensor(object_visibilities, dtype=torch.bool)
    if not isinstance(object_motions_valid, torch.Tensor):
        object_motions_valid = torch.tensor(object_motions_valid, dtype=torch.bool)

    for frame_idx in range(start_frame, end_frame):
        x = vertices[frame_idx]
        current_object_points = object_points[frame_idx]
        current_object_visibilities = object_visibilities[frame_idx]
        # The motion valid indicates if the tracking is valid from prev_frame
        current_object_motions_valid = object_motions_valid[frame_idx - 1]

        # Compute the single-direction chamfer loss for the object points
        chamfer_object_points = current_object_points[current_object_visibilities]
        chamfer_x = x[:num_surface_points]
        # The GT chamfer_object_points can be partial,first find the nearest in second
        chamfer_error = chamfer_distance(
            chamfer_object_points.unsqueeze(0),
            chamfer_x.unsqueeze(0),
            single_directional=True,
            norm=1,  # Get the L1 distance
        )[0]

        chamfer_errors.append(chamfer_error.item())

    chamfer_errors = np.array(chamfer_errors)

    results = {
        "frame_len": len(chamfer_errors),
        "chamfer_error": np.mean(chamfer_errors),
    }

    return results


def evaluate_case(case_name, exp_dir):
    """Evaluate a single experiment directory. Returns dict with train/test results or None."""
    inference_path = f"{exp_dir}/inference.pkl"
    data_path = f"{base_path}/{case_name}/final_data.pkl"
    split_path = f"{base_path}/{case_name}/split.json"

    if not os.path.exists(inference_path):
        print(f"  Skipping {exp_dir}: inference.pkl not found")
        return None
    if not os.path.exists(data_path):
        print(f"  Skipping {exp_dir}: final_data.pkl not found")
        return None
    if not os.path.exists(split_path):
        print(f"  Skipping {exp_dir}: split.json not found")
        return None

    with open(inference_path, "rb") as f:
        vertices = pickle.load(f)

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    object_points = data["object_points"]
    object_visibilities = data["object_visibilities"]
    object_motions_valid = data["object_motions_valid"]
    num_original_points = object_points.shape[1]
    num_surface_points = num_original_points + data["surface_points"].shape[0]

    with open(split_path, "r") as f:
        split = json.load(f)
    train_frame = split["train"][1]
    test_frame = split["test"][1]

    if test_frame != vertices.shape[0]:
        print(f"  Warning: test_frame {test_frame} != vertices frames {vertices.shape[0]}")
        test_frame = min(test_frame, vertices.shape[0])

    results_train = evaluate_prediction(
        1, train_frame, vertices, object_points, object_visibilities,
        object_motions_valid, num_original_points, num_surface_points,
    )
    results_test = evaluate_prediction(
        train_frame, test_frame, vertices, object_points, object_visibilities,
        object_motions_valid, num_original_points, num_surface_points,
    )

    return {
        "train_frames": results_train["frame_len"],
        "train_chamfer": results_train["chamfer_error"],
        "test_frames": results_test["frame_len"],
        "test_chamfer": results_test["chamfer_error"],
    }


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
    all_results = []  # list of (case_name, variant, train_frames, train_chamfer, test_frames, test_chamfer)

    file = open(output_file, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(file)
    writer.writerow([
        "Case Name", "Variant",
        "Train Frame Num", "Train Chamfer Error",
        "Test Frame Num", "Test Chamfer Error",
    ])

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
            exp_dir = f"{prediction_dir}/{case_name}{suffix}"
            result = evaluate_case(case_name, exp_dir)
            if result is not None:
                writer.writerow([
                    case_name, variant_name,
                    result["train_frames"], result["train_chamfer"],
                    result["test_frames"], result["test_chamfer"],
                ])
                all_results.append((case_name, variant_name, result["train_chamfer"], result["test_chamfer"]))
                print(f"  {variant_name:<10} -> train: {result['train_chamfer']:.6f}, test: {result['test_chamfer']:.6f}")

    file.close()
    print(f"\nResults saved to {output_file}")

    # ── Plotting ──
    if not all_results:
        print("No results to plot.")
        exit()

    # Group results by case_name
    case_order = []
    seen = set()
    for case_name, _, _, _ in all_results:
        if case_name not in seen:
            case_order.append(case_name)
            seen.add(case_name)

    variants = []
    if args.normal: variants.append("normal")
    if args.plasticity: variants.append("plasticity")
    if args.breakage: variants.append("breakage")

    # Prepare data
    data = {"Train": {v: {} for v in variants}, "Test": {v: {} for v in variants}}
    for case_name, variant, train_err, test_err in all_results:
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
                    ax.text(pos[k], v + max(max(vals), 1e-5)*0.02, f"{v:.4f}", ha="center", va="bottom", fontsize=8, rotation=0)

    ax.set_xlabel("Case", fontweight="bold")
    ax.set_ylabel("Chamfer Error (mm)", fontweight="bold")
    ax.set_title("Chamfer Error", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(case_order, rotation=0, ha="center")
    
    for i in range(len(case_order) - 1):
        ax.axvline(x[i] + 0.5, color='gray', linestyle='--', alpha=0.3)
        
    ax.legend(bbox_to_anchor=(0, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig("results/chamfer_error_comparison.png", dpi=150, bbox_inches="tight")
    print("Plot saved to results/chamfer_error_comparison.png")
    plt.show()
