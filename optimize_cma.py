# The first stage to optimize the sparse parameters using CMA-ES
from qqtt import OptimizerCMA
from qqtt.utils import logger, cfg
from qqtt.utils.logger import StreamToLogger, logging
import random
import numpy as np
import sys
import os
import torch
import pickle
import json
from argparse import ArgumentParser


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--train_frame", type=int, default=None)
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument("--hardening_factor", type=float, default=None)
    parser.add_argument("--enable_plasticity", action="store_true")
    parser.add_argument("--enable_breakage", action="store_true")
    parser.add_argument("--break_strain", type=float, default=None)
    parser.add_argument("--sim_method", type=str, default="spring_mass", choices=["spring_mass", "xpbd"])
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name
    train_frame = args.train_frame
    max_iter = args.max_iter

    # If train_frame not provided, read it from split.json
    if train_frame is None:
        split_path = f"{base_path}/{case_name}/split.json"
        assert os.path.exists(split_path), f"--train_frame not provided and split.json not found at {split_path}"
        with open(split_path, "r") as f:
            split_data = json.load(f)
        train_frame = split_data["train"][1]
        logger.info(f"train_frame not provided, read from split.json: {train_frame}")

    if "cloth" in case_name or "package" in case_name:
        cfg.load_from_yaml("configs/cloth.yaml")
    else:
        cfg.load_from_yaml("configs/real.yaml")

    # Set the intrinsic and extrinsic parameters for visualization
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
    else:
        # Fallback for old data if needed, or specific logic
        pass

    # Override config with command line arguments
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

    # Build experiment directory suffix based on enabled features
    suffix = ""
    if cfg.enable_plasticity:
        suffix += "_ep"
    if cfg.enable_breakage:
        suffix += "_brk"
    base_dir = f"experiments_optimization/{case_name}{suffix}"

    logger.set_log_file(path=base_dir, name="optimize_cma_log")
    optimizer = OptimizerCMA(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        train_frame=train_frame,
    )
    optimizer.optimize(max_iter=max_iter)
