"""
Unified ChipTwin entry point.

Branches between the original CMA-ES + Adam + Warp pipeline and the
optional GNN world model based on config flags.

Usage examples
--------------
    # Original pipeline (default – unchanged)
    python main.py --base_path ./data/different_types --case_name demo_cable

    # GNN data generation
    python main.py --base_path ./data/different_types --case_name demo_cable \\
        --use_gnn_world_model --gnn_generate_data

    # GNN offline training
    python main.py --use_gnn_world_model --gnn_train

    # GNN fast inference
    python main.py --base_path ./data/different_types --case_name demo_cable \\
        --use_gnn_world_model

    # GNN online fine-tuning
    python main.py --base_path ./data/different_types --case_name demo_cable \\
        --use_gnn_world_model --gnn_online_finetune
"""

from __future__ import annotations

import os
import sys
import json
from argparse import ArgumentParser


def _build_extra_flags(args) -> str:
    """Build shared optional flag string from parsed args."""
    flags = ""
    if getattr(args, "enable_plasticity", False):
        flags += " --enable_plasticity"
    if getattr(args, "enable_breakage", False):
        flags += " --enable_breakage"
    if getattr(args, "hardening_factor", None) is not None:
        flags += f" --hardening_factor {args.hardening_factor}"
    if getattr(args, "break_strain", None) is not None:
        flags += f" --break_strain {args.break_strain}"
    return flags


def run_original_pipeline(args) -> None:
    """Run the original CMA-ES + Adam + Warp pipeline (unchanged).

    This simply delegates to ``script_optimize_and_train.py`` or the
    individual ``train_warp.py`` / ``inference_warp.py`` scripts.
    """
    base_path = args.base_path
    case_name = args.case_name
    extra = _build_extra_flags(args)

    train_frame = getattr(args, "train_frame", None)
    if train_frame is None:
        split_path = f"{base_path}/{case_name}/split.json"
        if os.path.exists(split_path):
            with open(split_path, "r") as f:
                split_data = json.load(f)
            train_frame = split_data["train"][1]
            print(f"[INFO] train_frame from split.json: {train_frame}")

    # Step 1: optimize
    if not getattr(args, "skip_optimize", False):
        cmd = (
            f"python optimize_cma.py --base_path {base_path} --case_name {case_name}"
            f" --train_frame {train_frame} --max_iter {getattr(args, 'max_iter', 20)}"
            f"{extra}"
        )
        print(f"\n[STEP 1] {cmd}\n")
        ret = os.system(cmd)
        if ret != 0:
            sys.exit(ret)

    # Step 2: train
    if not getattr(args, "skip_train", False):
        cmd = (
            f"python train_warp.py --base_path {base_path} --case_name {case_name}"
            f" --train_frame {train_frame}{extra}"
        )
        print(f"\n[STEP 2] {cmd}\n")
        ret = os.system(cmd)
        if ret != 0:
            sys.exit(ret)

    # Step 3: inference
    if not getattr(args, "skip_inference", False):
        cmd = (
            f"python inference_warp.py --base_path {base_path} --case_name {case_name}"
            f"{extra}"
        )
        print(f"\n[STEP 3] {cmd}\n")
        ret = os.system(cmd)
        if ret != 0:
            sys.exit(ret)

    print(f"\n[DONE] Original pipeline finished for '{case_name}'.")


def main() -> None:
    """Parse arguments and dispatch to the correct pipeline."""
    parser = ArgumentParser(description="ChipTwin unified entry point.")

    # Common
    parser.add_argument("--config", type=str, default="configs/real.yaml")
    parser.add_argument("--base_path", type=str, default="./data/different_types")
    parser.add_argument("--case_name", type=str, default="")
    parser.add_argument("--train_frame", type=int, default=None)
    parser.add_argument("--enable_plasticity", action="store_true")
    parser.add_argument("--enable_breakage", action="store_true")
    parser.add_argument("--hardening_factor", type=float, default=None)
    parser.add_argument("--break_strain", type=float, default=None)

    # Original pipeline flags
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument("--skip_optimize", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_inference", action="store_true")

    # GNN flags
    parser.add_argument("--use_gnn_world_model", action="store_true")
    parser.add_argument("--gnn_generate_data", action="store_true")
    parser.add_argument("--gnn_train", action="store_true")
    parser.add_argument("--gnn_online_finetune", action="store_true")
    parser.add_argument("--gnn_inference", action="store_true")
    parser.add_argument("--gnn_num_trajectories", type=int, default=None)
    parser.add_argument("--gnn_output_dir", type=str, default=None)
    parser.add_argument("--gnn_data_dir", type=str, default=None)
    parser.add_argument("--gnn_checkpoint_dir", type=str, default=None)
    parser.add_argument("--gnn_epochs", type=int, default=None)
    parser.add_argument("--gnn_lr", type=float, default=None)
    parser.add_argument("--gnn_hidden_dim", type=int, default=None)
    parser.add_argument("--gnn_message_passing_steps", type=int, default=None)
    parser.add_argument("--gnn_finetune_frames", type=int, default=None)
    parser.add_argument("--gnn_finetune_steps", type=int, default=None)
    parser.add_argument("--gnn_finetune_lr", type=float, default=None)

    args = parser.parse_args()

    # ── Load config ──
    from qqtt.utils import cfg
    cfg.load_from_yaml(args.config)

    # Override GNN config from CLI
    for attr in [
        "gnn_num_trajectories", "gnn_output_dir", "gnn_data_dir",
        "gnn_checkpoint_dir", "gnn_epochs", "gnn_lr", "gnn_hidden_dim",
        "gnn_message_passing_steps", "gnn_finetune_frames",
        "gnn_finetune_steps", "gnn_finetune_lr",
    ]:
        val = getattr(args, attr, None)
        if val is not None:
            setattr(cfg, attr, val)

    if args.enable_plasticity:
        cfg.enable_plasticity = True
    if args.enable_breakage:
        cfg.enable_breakage = True
    cfg.device = "cuda:0"

    # ── Branch ──
    if args.use_gnn_world_model:
        if args.gnn_generate_data:
            from gnn.generate_data import run_gnn_data_generation
            run_gnn_data_generation(args)
        elif args.gnn_train:
            from gnn.train_gnn import run_gnn_training
            run_gnn_training(args)
        elif args.gnn_online_finetune:
            from gnn.online_finetune import run_gnn_online_finetune
            run_gnn_online_finetune(args)
        elif args.gnn_inference:
            from gnn.inference import run_gnn_inference
            run_gnn_inference(args)
        else:
            # Default: run inference when --use_gnn_world_model is set
            from gnn.inference import run_gnn_inference
            run_gnn_inference(args)
    else:
        run_original_pipeline(args)


if __name__ == "__main__":
    main()
