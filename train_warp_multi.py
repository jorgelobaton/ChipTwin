"""
Multi-experiment training entry point.

Learns shared material parameters (break_strain, yield_strain, collide_*, etc.)
across multiple experiments so that breakage is observed in some but not all.

Usage:
    python train_warp_multi.py \
        --base_path ./data/different_types \
        --case_names demo_70,demo_58_new,single_push_rope_1 \
        --enable_breakage \
        --comment "multi-experiment breakage learning"

Each case_name must have:
  - final_data.pkl, calibrate.pkl, metadata.json, split.json  under base_path/case_name
  - optimal_params.pkl  under experiments_optimization/case_name{suffix}/
"""

from qqtt.engine.trainer_warp_multi import InvPhyTrainerMulti, ExperimentData
from qqtt.utils import logger, cfg
from datetime import datetime
import random
import numpy as np
import torch
from argparse import ArgumentParser
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
    parser.add_argument(
        "--base_path", type=str, required=True,
        help="Base directory containing all experiment case folders",
    )
    parser.add_argument(
        "--case_names", type=str, required=True,
        help="Comma-separated list of case names to train on jointly",
    )
    parser.add_argument("--hardening_factor", type=float, default=None)
    parser.add_argument("--enable_plasticity", action="store_true")
    parser.add_argument("--enable_breakage", action="store_true")
    parser.add_argument("--break_strain", type=float, default=None)
    parser.add_argument(
        "--sim_method", type=str, default="spring_mass",
        choices=["spring_mass", "xpbd"],
    )
    parser.add_argument("--chamfer_weight", type=float, default=None)
    parser.add_argument("--track_weight", type=float, default=None)
    parser.add_argument("--acc_weight", type=float, default=None)
    parser.add_argument(
        "--comment", type=str, default=None,
        help="Comment to tag the wandb run",
    )
    parser.add_argument(
        "--group_name", type=str, default=None,
        help="Override the output directory name (default: auto-generated from case names)",
    )
    args = parser.parse_args()

    base_path = args.base_path
    case_names = [c.strip() for c in args.case_names.split(",")]
    assert len(case_names) >= 1, "Provide at least one case name"

    # Load config YAML (use the first case's type to pick config)
    first_case = case_names[0]
    if "cloth" in first_case or "package" in first_case:
        config_yaml_path = "configs/cloth.yaml"
    else:
        config_yaml_path = "configs/real.yaml"
    cfg.load_from_yaml(config_yaml_path)
    cfg.config_yaml_path = os.path.abspath(config_yaml_path)

    print(f"[DATA TYPE]: {cfg.data_type}")
    print(f"[EXPERIMENTS]: {case_names}")

    # Build suffix
    suffix = ""
    if args.enable_plasticity:
        suffix += "_ep"
    if args.enable_breakage:
        suffix += "_brk"

    # Override config
    if args.hardening_factor is not None:
        cfg.hardening_factor = args.hardening_factor
    if args.enable_plasticity:
        cfg.enable_plasticity = True
    if args.enable_breakage:
        cfg.enable_breakage = True
    if args.break_strain is not None:
        cfg.break_strain = args.break_strain
    if args.sim_method:
        cfg.sim_method = args.sim_method
    if args.chamfer_weight is not None:
        cfg.chamfer_weight = args.chamfer_weight
    if args.track_weight is not None:
        cfg.track_weight = args.track_weight
    if args.acc_weight is not None:
        cfg.acc_weight = args.acc_weight
    cfg.comment = args.comment

    # Load per-experiment data
    experiment_data_list = []
    for case_name in case_names:
        # Load optimal params
        optimal_path = f"experiments_optimization/{case_name}{suffix}/optimal_params.pkl"
        assert os.path.exists(optimal_path), (
            f"{case_name}: Optimal parameters not found: {optimal_path}"
        )
        with open(optimal_path, "rb") as f:
            optimal_params = pickle.load(f)

        # Load split to get train_frame
        split_path = f"{base_path}/{case_name}/split.json"
        assert os.path.exists(split_path), f"{case_name}: split.json not found: {split_path}"
        with open(split_path, "r") as f:
            split = json.load(f)
        train_frame = split["train"][1]

        experiment_data_list.append(
            ExperimentData(case_name, base_path, optimal_params, train_frame)
        )

    # Build output directory name
    if args.group_name:
        group_label = args.group_name
    else:
        group_label = "multi_" + "_".join(case_names)

    base_dir = f"experiments/{group_label}{suffix}"

    logger.set_log_file(path=base_dir, name="inv_phy_multi_log")
    trainer = InvPhyTrainerMulti(
        experiment_data_list=experiment_data_list,
        base_dir=base_dir,
    )
    trainer.train()
