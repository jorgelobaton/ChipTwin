"""
Multi-experiment CMA-ES optimization entry point.

Optimizes shared global parameters across multiple experiments so that the
resulting params work well for diverse deformation scenarios.

Usage:
    python optimize_cma_multi.py \
        --base_path ./data/different_types \
        --case_names demo_70,demo_58_new,single_push_rope_1 \
        --enable_breakage \
        --max_iter 20
"""

from qqtt.engine.cma_optimize_warp_multi import OptimizerCMAMulti
from qqtt.utils import logger, cfg
from qqtt.utils.logger import StreamToLogger, logging
import random
import numpy as np
import sys
import torch
import pickle
import json
from argparse import ArgumentParser


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

sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument(
        "--case_names", type=str, required=True,
        help="Comma-separated list of case names",
    )
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument("--hardening_factor", type=float, default=None)
    parser.add_argument("--enable_plasticity", action="store_true")
    parser.add_argument("--enable_breakage", action="store_true")
    parser.add_argument("--break_strain", type=float, default=None)
    parser.add_argument(
        "--sim_method", type=str, default="spring_mass",
        choices=["spring_mass", "xpbd"],
    )
    parser.add_argument(
        "--group_name", type=str, default=None,
        help="Override the output directory name",
    )
    args = parser.parse_args()

    base_path = args.base_path
    case_names = [c.strip() for c in args.case_names.split(",")]
    assert len(case_names) >= 1, "Provide at least one case name"

    # Load config YAML
    first_case = case_names[0]
    if "cloth" in first_case or "package" in first_case:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

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

    suffix = ""
    if cfg.enable_plasticity:
        suffix += "_ep"
    if cfg.enable_breakage:
        suffix += "_brk"

    # Build experiment configs
    experiment_configs = []
    for case_name in case_names:
        # Load calibration into cfg for visualization (use first experiment's)
        with open(f"{base_path}/{case_name}/calibrate.pkl", "rb") as f:
            c2ws = pickle.load(f)
        w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
        with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
            metadata = json.load(f)

        # Load split for train_frame
        with open(f"{base_path}/{case_name}/split.json", "r") as f:
            split = json.load(f)

        experiment_configs.append({
            "case_name": case_name,
            "base_path": base_path,
            "data_path": f"{base_path}/{case_name}/final_data.pkl",
            "train_frame": split["train"][1],
        })

    # Use first experiment's camera config for cfg (needed by visualize_pc)
    first_case_name = case_names[0]
    with open(f"{base_path}/{first_case_name}/calibrate.pkl", "rb") as f:
        c2ws = pickle.load(f)
    w2cs = [np.linalg.inv(c2w) for c2w in c2ws]
    cfg.c2ws = np.array(c2ws)
    cfg.w2cs = np.array(w2cs)
    with open(f"{base_path}/{first_case_name}/metadata.json", "r") as f:
        data = json.load(f)
    cfg.intrinsics = np.array(data["intrinsics"])
    cfg.WH = data["WH"]
    cfg.overlay_path = f"{base_path}/{first_case_name}/color"
    if "camera_ids" in data:
        cfg.camera_ids = data["camera_ids"]

    # Build output directory
    if args.group_name:
        group_label = args.group_name
    else:
        group_label = "multi_" + "_".join(case_names)
    base_dir = f"experiments_optimization/{group_label}{suffix}"

    logger.set_log_file(path=base_dir, name="optimize_cma_multi_log")
    optimizer = OptimizerCMAMulti(
        experiment_configs=experiment_configs,
        base_dir=base_dir,
    )
    optimizer.optimize(max_iter=args.max_iter)
