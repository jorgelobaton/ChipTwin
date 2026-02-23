"""Run optimize_cma followed by train_warp for a single case.

Usage examples:
    python script_optimize_and_train.py --base_path ./data/different_types --case_name demo_64
    python script_optimize_and_train.py --base_path ./data/different_types --case_name demo_64 --enable_plasticity
    python script_optimize_and_train.py --base_path ./data/different_types --case_name demo_64 --enable_plasticity --enable_breakage
    python script_optimize_and_train.py --base_path ./data/different_types --case_name demo_70 --run_gs
    python script_optimize_and_train.py --base_path ./data/different_types --case_name demo_70 --run_gs --gs_iterations 10000 --gs_lambda_depth 0.001 --gs_lambda_seg 1.0
"""
import os
import sys
import json
from argparse import ArgumentParser


def build_extra_flags(args):
    """Build the shared optional flags string from parsed args."""
    flags = ""
    if args.enable_plasticity:
        flags += " --enable_plasticity"
    if args.enable_breakage:
        flags += " --enable_breakage"
    if args.hardening_factor is not None:
        flags += f" --hardening_factor {args.hardening_factor}"
    if args.break_strain is not None:
        flags += f" --break_strain {args.break_strain}"
    if args.sim_method != "spring_mass":
        flags += f" --sim_method {args.sim_method}"
    if args.plasticity_smooth_weight is not None:
        flags += f" --plasticity_smooth_weight {args.plasticity_smooth_weight}"
    if args.plasticity_init_noise is not None:
        flags += f" --plasticity_init_noise {args.plasticity_init_noise}"
    return flags


if __name__ == "__main__":
    parser = ArgumentParser(description="Run optimize_cma then train_warp for a single case.")
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--train_frame", type=int, default=None,
                        help="Number of training frames. If omitted, read from split.json.")
    # optimize_cma-specific
    parser.add_argument("--max_iter", type=int, default=20,
                        help="Max CMA-ES iterations (optimize step).")
    # train_warp-specific
    parser.add_argument("--chamfer_weight", type=float, default=None)
    parser.add_argument("--track_weight", type=float, default=None)
    parser.add_argument("--acc_weight", type=float, default=None)
    parser.add_argument("--comment", type=str, default=None,
                        help="Comment to tag the wandb run.")
    # Shared flags
    parser.add_argument("--hardening_factor", type=float, default=None)
    parser.add_argument("--enable_plasticity", action="store_true")
    parser.add_argument("--enable_breakage", action="store_true")
    parser.add_argument("--break_strain", type=float, default=None)
    parser.add_argument("--sim_method", type=str, default="spring_mass",
                        choices=["spring_mass", "xpbd"])
    parser.add_argument("--plasticity_smooth_weight", type=float, default=None)
    parser.add_argument("--plasticity_init_noise", type=float, default=None)

    # Pipeline switches
    parser.add_argument("--skip_optimize", action="store_true", help="Skip optimize_cma step")
    parser.add_argument("--skip_train", action="store_true", help="Skip train_warp step")
    parser.add_argument("--skip_inference", action="store_true", help="Skip inference_warp step")

    # Gaussian Splatting steps (opt-in)
    parser.add_argument("--run_gs", action="store_true",
                        help="Also run export_gaussian_data and gs_train before optimize/train.")
    parser.add_argument("--gs_iterations", type=int, default=10000,
                        help="Number of gs_train iterations (default: 10000).")
    parser.add_argument("--gs_lambda_depth", type=float, default=0.001)
    parser.add_argument("--gs_lambda_normal", type=float, default=0.0)
    parser.add_argument("--gs_lambda_anisotropic", type=float, default=0.0)
    parser.add_argument("--gs_lambda_seg", type=float, default=1.0)
    parser.add_argument("--gs_init_opt", type=str, default="hybrid",
                        help="gs_train --gs_init_opt value (default: hybrid).")
    parser.add_argument("--gs_no_isotropic", action="store_true",
                        help="Disable --isotropic flag in gs_train (isotropic is on by default).")
    parser.add_argument("--gs_no_masks", action="store_true",
                        help="Disable --use_masks flag in gs_train (use_masks is on by default).")
    args = parser.parse_args()

    base_path = args.base_path
    case_name = args.case_name
    train_frame = args.train_frame

    # If train_frame not provided, read from split.json
    if train_frame is None:
        split_path = f"{base_path}/{case_name}/split.json"
        assert os.path.exists(split_path), (
            f"--train_frame not provided and split.json not found at {split_path}"
        )
        with open(split_path, "r") as f:
            split_data = json.load(f)
        train_frame = split_data["train"][1]
        print(f"[INFO] train_frame not provided, read from split.json: {train_frame}")

    extra_flags = build_extra_flags(args)

    # ── Step 0: export_gaussian_data ─────────────────────────────────────────
    if args.run_gs:
        export_cmd = (
            f"python export_gaussian_data.py"
            f" --case_name {case_name}"
        )
        print(f"\n[STEP 0] Running: {export_cmd}\n")
        ret = os.system(export_cmd)
        if ret != 0:
            print(f"[ERROR] export_gaussian_data.py failed with exit code {ret}. Aborting.")
            sys.exit(ret)

        # ── Step 0.5: gs_train ───────────────────────────────────────────────
        gs_model_path = (
            f"./gaussian_output/{case_name}/"
            f"init={args.gs_init_opt}"
            f"_iso={not args.gs_no_isotropic}"
            f"_ldepth={args.gs_lambda_depth}"
            f"_lnormal={args.gs_lambda_normal}"
            f"_laniso_{args.gs_lambda_anisotropic}"
            f"_lseg={args.gs_lambda_seg}"
        )
        gs_train_cmd = (
            f"python gs_train.py"
            f" -s ./data/gaussian_data/{case_name}"
            f" -m {gs_model_path}"
            f" --iterations {args.gs_iterations}"
            f" --lambda_depth {args.gs_lambda_depth}"
            f" --lambda_normal {args.gs_lambda_normal}"
            f" --lambda_anisotropic {args.gs_lambda_anisotropic}"
            f" --lambda_seg {args.gs_lambda_seg}"
            f" --gs_init_opt '{args.gs_init_opt}'"
        )
        if not args.gs_no_isotropic:
            gs_train_cmd += " --isotropic"
        if not args.gs_no_masks:
            gs_train_cmd += " --use_masks"
        print(f"\n[STEP 0.5] Running: {gs_train_cmd}\n")
        ret = os.system(gs_train_cmd)
        if ret != 0:
            print(f"[ERROR] gs_train.py failed with exit code {ret}. Aborting.")
            sys.exit(ret)
    else:
        print("\n[STEP 0 / 0.5] Skipping export_gaussian_data + gs_train (use --run_gs to enable).\n")

    # ── Step 1: optimize ─────────────────────────────────────────────────────
    if not args.skip_optimize:
        optimize_cmd = (
            f"python optimize_cma.py"
            f" --base_path {base_path}"
            f" --case_name {case_name}"
            f" --train_frame {train_frame}"
            f" --max_iter {args.max_iter}"
            f"{extra_flags}"
        )
        print(f"\n[STEP 1] Running: {optimize_cmd}\n")
        ret = os.system(optimize_cmd)
        if ret != 0:
            print(f"[ERROR] optimize_cma.py failed with exit code {ret}. Aborting.")
            sys.exit(ret)
    else:
        print("\n[STEP 1] Skipping optimize_cma.py (requested).\n")

    # ── Step 2: train ─────────────────────────────────────────────────────────
    if not args.skip_train:
        train_extra = extra_flags
        if args.chamfer_weight is not None:
            train_extra += f" --chamfer_weight {args.chamfer_weight}"
        if args.track_weight is not None:
            train_extra += f" --track_weight {args.track_weight}"
        if args.acc_weight is not None:
            train_extra += f" --acc_weight {args.acc_weight}"
        if args.comment is not None:
            train_extra += f" --comment \"{args.comment}\""

        train_cmd = (
            f"python train_warp.py"
            f" --base_path {base_path}"
            f" --case_name {case_name}"
            f" --train_frame {train_frame}"
            f"{train_extra}"
        )
        print(f"\n[STEP 2] Running: {train_cmd}\n")
        ret = os.system(train_cmd)
        if ret != 0:
            print(f"[ERROR] train_warp.py failed with exit code {ret}.")
            sys.exit(ret)
    else:
        print("\n[STEP 2] Skipping train_warp.py (requested).\n")

    # ── Step 3: inference ─────────────────────────────────────────────────────
    if not args.skip_inference:
        inference_cmd = (
            f"python inference_warp.py"
            f" --base_path {base_path}"
            f" --case_name {case_name}"
            f"{extra_flags}"
        )
        print(f"\n[STEP 3] Running: {inference_cmd}\n")
        ret = os.system(inference_cmd)
        if ret != 0:
            print(f"[ERROR] inference_warp.py failed with exit code {ret}.")
            sys.exit(ret)
    else:
        print("\n[STEP 3] Skipping inference_warp.py (requested).\n")

    print(f"\n[DONE] Pipeline finished for case '{case_name}'.")
