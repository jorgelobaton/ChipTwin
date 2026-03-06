"""
Inference entry point for multi-experiment checkpoints.

Runs inference on each experiment using the shared global params from a
multi-experiment checkpoint + per-experiment spring_Y.

Usage:
    python inference_warp_multi.py \
        --base_path ./data/different_types \
        --case_names demo_70,demo_58_new \
        --enable_breakage \
        --group_name multi_demo_70_demo_58_new
"""

from qqtt import InvPhyTrainerWarp
from qqtt.utils import logger, cfg
import random
import numpy as np
import torch
from argparse import ArgumentParser
import glob
import os
import pickle
import json


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_names", type=str, required=True,
                        help="Comma-separated list of case names to run inference on")
    parser.add_argument("--enable_plasticity", action="store_true")
    parser.add_argument("--enable_breakage", action="store_true")
    parser.add_argument("--break_strain", type=float, default=None)
    parser.add_argument("--group_name", type=str, default=None,
                        help="Name of the multi-experiment group (must match training)")
    args = parser.parse_args()

    base_path = args.base_path
    case_names = [c.strip() for c in args.case_names.split(",")]

    suffix = ""
    if args.enable_plasticity:
        suffix += "_ep"
    if args.enable_breakage:
        suffix += "_brk"

    if args.group_name:
        group_label = args.group_name
    else:
        group_label = "multi_" + "_".join(case_names)

    multi_base_dir = f"experiments/{group_label}{suffix}"

    # Find best model
    best_models = glob.glob(f"{multi_base_dir}/train/best_*.pth")
    assert len(best_models) > 0, f"No best model found in {multi_base_dir}/train/"
    best_model_path = best_models[0]
    print(f"Using multi-experiment checkpoint: {best_model_path}")

    for case_name in case_names:
        print(f"\n{'='*60}")
        print(f"Running inference for: {case_name}")
        print(f"{'='*60}")

        if "cloth" in case_name or "package" in case_name:
            cfg.load_from_yaml("configs/cloth.yaml")
        else:
            cfg.load_from_yaml("configs/real.yaml")

        # Load optimal params for this experiment's topology
        optimal_path = f"experiments_optimization/{case_name}{suffix}/optimal_params.pkl"
        if not os.path.exists(optimal_path):
            # Try multi-experiment optimal params
            multi_opt_path = f"experiments_optimization/{group_label}{suffix}/optimal_params.pkl"
            assert os.path.exists(multi_opt_path), (
                f"Neither {optimal_path} nor {multi_opt_path} found"
            )
            optimal_path = multi_opt_path

        with open(optimal_path, "rb") as f:
            optimal_params = pickle.load(f)
        cfg.set_optimal_params(optimal_params)

        # Camera config
        with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
            c2ws = pickle.load(f)
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
        cfg.c2ws = np.array(c2ws)
        cfg.w2cs = np.array(w2cs)
        with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
            data = json.load(f)
        cfg.intrinsics = np.array(data["intrinsics"])
        cfg.WH = data["WH"]
        cfg.overlay_path = f"{base_path}/{case_name}/color"
        if "camera_ids" in data:
            cfg.camera_ids = data["camera_ids"]

        if args.enable_plasticity:
            cfg.enable_plasticity = True
        if args.enable_breakage:
            cfg.enable_breakage = True
        if args.break_strain is not None:
            cfg.break_strain = args.break_strain

        # Output goes to per-experiment subdir under the multi dir
        exp_base_dir = f"{multi_base_dir}/inference/{case_name}"
        cfg.base_dir = exp_base_dir
        os.makedirs(exp_base_dir, exist_ok=True)

        logger.set_log_file(path=exp_base_dir, name="inference_log")
        trainer = InvPhyTrainerWarp(
            data_path=f"{base_path}/{case_name}/final_data.pkl",
            base_dir=exp_base_dir,
            pure_inference_mode=True,
        )
        trainer.test(best_model_path, case_name=case_name)

    print(f"\nAll inference complete. Results in: {multi_base_dir}/inference/")
