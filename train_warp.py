from qqtt import InvPhyTrainerWarp
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
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed = 42
set_all_seeds(seed)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--train_frame", type=int, required=True)
    parser.add_argument("--hardening_factor", type=float, default=None)
    parser.add_argument("--enable_plasticity", action="store_true")
    parser.add_argument("--enable_breakage", action="store_true")
    parser.add_argument("--break_strain", type=float, default=None)
    parser.add_argument("--sim_method", type=str, default="spring_mass", choices=["spring_mass", "xpbd"])
    parser.add_argument("--chamfer_weight", type=float, default=None)
    parser.add_argument("--track_weight", type=float, default=None)
    parser.add_argument("--acc_weight", type=float, default=None)
    parser.add_argument("--comment", type=str, default=None, help="Comment to tag the wandb run for easier identification")
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name
    train_frame = args.train_frame

    if "cloth" in case_name or "package" in case_name:
        config_yaml_path = "configs/cloth.yaml"
    else:
        config_yaml_path = "configs/real.yaml"
    cfg.load_from_yaml(config_yaml_path)
    cfg.config_yaml_path = os.path.abspath(config_yaml_path)

    print(f"[DATA TYPE]: {cfg.data_type}")

    # Build experiment directory suffix based on enabled features
    suffix = ""
    if args.enable_plasticity:
        suffix += "_ep"
    if args.enable_breakage:
        suffix += "_brk"

    # Read the first-satage optimized parameters
    optimal_path = f"experiments_optimization/{case_name}{suffix}/optimal_params.pkl"
    assert os.path.exists(
        optimal_path
    ), f"{case_name}: Optimal parameters not found: {optimal_path}"
    with open(optimal_path, "rb") as f:
        optimal_params = pickle.load(f)
    cfg.set_optimal_params(optimal_params)

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
    if args.chamfer_weight is not None:
        cfg.chamfer_weight = args.chamfer_weight
    if args.track_weight is not None:
        cfg.track_weight = args.track_weight
    if args.acc_weight is not None:
        cfg.acc_weight = args.acc_weight

    cfg.comment = args.comment

    base_dir = f"experiments/{case_name}{suffix}"

    logger.set_log_file(path=base_dir, name="inv_phy_log")
    trainer = InvPhyTrainerWarp(
        data_path=f"{base_path}/{case_name}/final_data.pkl",
        base_dir=base_dir,
        train_frame=train_frame,
    )
    trainer.train()
