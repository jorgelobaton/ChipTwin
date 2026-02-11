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
    # Read case names from data_config.csv
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

        # Evaluate normal variant
        normal_dir = f"{prediction_dir}/{case_name}"
        result_normal = evaluate_case(case_name, normal_dir)
        if result_normal is not None:
            writer.writerow([
                case_name, "normal",
                result_normal["train_frames"], result_normal["train_chamfer"],
                result_normal["test_frames"], result_normal["test_chamfer"],
            ])
            all_results.append((case_name, "normal", result_normal["train_chamfer"], result_normal["test_chamfer"]))
            print(f"  normal  -> train: {result_normal['train_chamfer']:.6f}, test: {result_normal['test_chamfer']:.6f}")

        # Evaluate _ep (elastoplastic) variant
        ep_dir = f"{prediction_dir}/{case_name}_ep"
        result_ep = evaluate_case(case_name, ep_dir)
        if result_ep is not None:
            writer.writerow([
                case_name, "plasticity",
                result_ep["train_frames"], result_ep["train_chamfer"],
                result_ep["test_frames"], result_ep["test_chamfer"],
            ])
            all_results.append((case_name, "plasticity", result_ep["train_chamfer"], result_ep["test_chamfer"]))
            print(f"  plastic -> train: {result_ep['train_chamfer']:.6f}, test: {result_ep['test_chamfer']:.6f}")

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

    normal_train = {}
    normal_test = {}
    ep_train = {}
    ep_test = {}
    for case_name, variant, train_err, test_err in all_results:
        if variant == "normal":
            normal_train[case_name] = train_err
            normal_test[case_name] = test_err
        else:
            ep_train[case_name] = train_err
            ep_test[case_name] = test_err

    x = np.arange(len(case_order))
    width = 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8, len(case_order) * 3), 6))

    # Train errors
    if normal_train:
        vals = [normal_train.get(c, 0) for c in case_order]
        has = [c in normal_train for c in case_order]
        ax1.bar(x - width / 2, vals, width, label="Normal", color="#4C72B0", alpha=0.85)
        for i, (v, h) in enumerate(zip(vals, has)):
            if h:
                ax1.text(i - width / 2, v + v * 0.02, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    if ep_train:
        vals = [ep_train.get(c, 0) for c in case_order]
        has = [c in ep_train for c in case_order]
        ax1.bar(x + width / 2, vals, width, label="Plasticity", color="#DD8452", alpha=0.85)
        for i, (v, h) in enumerate(zip(vals, has)):
            if h:
                ax1.text(i + width / 2, v + v * 0.02, f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    ax1.set_xlabel("Case")
    ax1.set_ylabel("Chamfer Error")
    ax1.set_title("Train Chamfer Error")
    ax1.set_xticks(x)
    ax1.set_xticklabels(case_order, rotation=45, ha="right")
    ax1.legend()

    # Test errors
    if normal_test:
        vals = [normal_test.get(c, 0) for c in case_order]
        ax2.bar(x - width / 2, vals, width, label="Normal", color="#4C72B0", alpha=0.85)
        for i, (v, h) in enumerate(zip(vals, [c in normal_test for c in case_order])):
            if h:
                ax2.text(i - width / 2, v + v * 0.02, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    if ep_test:
        vals = [ep_test.get(c, 0) for c in case_order]
        ax2.bar(x + width / 2, vals, width, label="Plasticity", color="#DD8452", alpha=0.85)
        for i, (v, h) in enumerate(zip(vals, [c in ep_test for c in case_order])):
            if h:
                ax2.text(i + width / 2, v + v * 0.02, f"{v:.4f}", ha="center", va="bottom", fontsize=7)

    ax2.set_xlabel("Case")
    ax2.set_ylabel("Chamfer Error")
    ax2.set_title("Test Chamfer Error")
    ax2.set_xticks(x)
    ax2.set_xticklabels(case_order, rotation=45, ha="right")
    ax2.legend()

    plt.suptitle("Chamfer Error: Normal vs Plasticity", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/chamfer_error_comparison.png", dpi=150, bbox_inches="tight")
    print("Plot saved to results/chamfer_error_comparison.png")
    plt.show()
