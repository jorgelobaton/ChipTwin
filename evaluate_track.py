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
    split_path = f"{base_path}/{case_name}/split.json"

    if not os.path.exists(inference_path):
        print(f"  Skipping {exp_dir}: inference.pkl not found")
        return None
    if not os.path.exists(gt_track_path):
        print(f"  Skipping {exp_dir}: gt_track_3d.pkl not found")
        return None
    if not os.path.exists(split_path):
        print(f"  Skipping {exp_dir}: split.json not found")
        return None

    with open(split_path, "r") as f:
        split = json.load(f)
    train_frame = split["train"][1]
    test_frame = split["test"][1]

    with open(inference_path, "rb") as f:
        vertices = pickle.load(f)

    with open(gt_track_path, "rb") as f:
        gt_track_3d = pickle.load(f)

    # Locate the index of corresponding point index in the vertices
    mask = ~np.isnan(gt_track_3d[0]).any(axis=1)
    kdtree = KDTree(vertices[0])
    _, idx = kdtree.query(gt_track_3d[0][mask])

    train_track_error = evaluate_prediction(1, train_frame, vertices, gt_track_3d, idx, mask)
    test_track_error = evaluate_prediction(train_frame, test_frame, vertices, gt_track_3d, idx, mask)

    return train_track_error, test_track_error


if __name__ == "__main__":
    # Read case names from data_config.csv
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

        # Evaluate normal variant
        normal_dir = f"{prediction_path}/{case_name}"
        result_normal = evaluate_case(case_name, normal_dir)
        if result_normal is not None:
            train_err, test_err = result_normal
            writer.writerow([case_name, "normal", train_err, test_err])
            results.append((case_name, "normal", train_err, test_err))
            print(f"  normal  -> train: {train_err:.6f}, test: {test_err:.6f}")

        # Evaluate _ep (elastoplastic) variant
        ep_dir = f"{prediction_path}/{case_name}_ep"
        result_ep = evaluate_case(case_name, ep_dir)
        if result_ep is not None:
            train_err, test_err = result_ep
            writer.writerow([case_name, "plasticity", train_err, test_err])
            results.append((case_name, "plasticity", train_err, test_err))
            print(f"  plastic -> train: {train_err:.6f}, test: {test_err:.6f}")

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

    # Prepare data for grouped bar chart
    normal_train = {}
    normal_test = {}
    ep_train = {}
    ep_test = {}
    for case_name, variant, train_err, test_err in results:
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
    ax1.set_ylabel("Track Error")
    ax1.set_title("Train Track Error")
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
    ax2.set_ylabel("Track Error")
    ax2.set_title("Test Track Error")
    ax2.set_xticks(x)
    ax2.set_xticklabels(case_order, rotation=45, ha="right")
    ax2.legend()

    plt.suptitle("Track Error: Normal vs Plasticity", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/track_error_comparison.png", dpi=150, bbox_inches="tight")
    print("Plot saved to results/track_error_comparison.png")
    plt.show()
