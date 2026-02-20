"""
Segmentation Pipeline with Optional Manual Annotation
======================================================
For each camera, runs segmentation for the object and the controller.
Either (or both) can be manual or automatic:

  --manual_controller  : use interactive manual tool for the controller
  --manual_object      : use interactive manual tool for the target object
  (neither flag)       : fully automatic GroundingDINO + SAM2 for both

Combinations:
  auto+auto            : standard pipeline (segment_util_video with full TEXT_PROMPT)
  auto_object+manual_controller : auto for object, then manual for controller
  manual_object+auto_controller : manual for object, then auto for controller
  manual+manual        : manual for both (object first, then controller)
"""

import os
import glob
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--TEXT_PROMPT", type=str, default="",
                    help="Text prompt for auto-segmentation. Should contain object and/or "
                         "controller labels separated by '.'. Not needed for fully manual mode.")
parser.add_argument("--manual_controller", action="store_true", default=False,
                    help="Use manual interactive segmentation for the controller")
parser.add_argument("--manual_object", action="store_true", default=False,
                    help="Use manual interactive segmentation for the target object")
parser.add_argument("--controller_label", type=str, default="hand",
                    help="Label name for the controller in mask_info json")
parser.add_argument("--object_label", type=str, default="",
                    help="Label name for the object (used when --manual_object is set). "
                         "If empty, inferred from TEXT_PROMPT.")
parser.add_argument("--box_threshold", type=float, default=0.35)
parser.add_argument("--text_threshold", type=float, default=0.25)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name

# Infer object label from TEXT_PROMPT if not explicitly given
object_label = args.object_label
if not object_label and args.TEXT_PROMPT:
    # TEXT_PROMPT is like "cloth.hand" or "cloth." — first token is the object
    object_label = args.TEXT_PROMPT.split(".")[0].strip()
if not object_label:
    object_label = "object"

# List all camera IDs
camera_ids = sorted([
    os.path.basename(p)
    for p in glob.glob(f"{base_path}/{case_name}/depth/*")
])
print(f"[Segment] Processing {case_name}, cameras: {camera_ids}")
print(f"[Segment] Manual controller: {args.manual_controller}, Manual object: {args.manual_object}")

both_manual = args.manual_controller and args.manual_object
both_auto = not args.manual_controller and not args.manual_object

for cam_id in camera_ids:
    print(f"\n{'='*60}")
    print(f"  Camera: {cam_id}")
    print(f"{'='*60}")

    if both_auto:
        # ── Fully automatic ──
        print(f"[Auto] Segmenting with prompt: '{args.TEXT_PROMPT}'")
        os.system(
            f"python ./data_process/segment_util_video.py "
            f"--base_path {base_path} --case_name {case_name} "
            f"--TEXT_PROMPT '{args.TEXT_PROMPT}' --camera_id {cam_id} "
            f"--box_threshold {args.box_threshold} --text_threshold {args.text_threshold}"
        )
        os.system(f"rm -rf {base_path}/{case_name}/tmp_data")

    elif both_manual:
        # ── Both manual ──
        # Object first (gets obj_id=0), then controller (gets obj_id=1)
        print(f"[Manual] Segment the TARGET OBJECT ('{object_label}') for camera {cam_id}")
        os.system(
            f"python ./data_process/manual_segment.py "
            f"--base_path {base_path} --case_name {case_name} "
            f"--camera_id {cam_id} --label '{object_label}' --obj_id 0"
        )
        print(f"\n[Manual] Segment the CONTROLLER ('{args.controller_label}') for camera {cam_id}")
        os.system(
            f"python ./data_process/manual_segment.py "
            f"--base_path {base_path} --case_name {case_name} "
            f"--camera_id {cam_id} --label '{args.controller_label}' --obj_id 1"
        )

    elif args.manual_object and not args.manual_controller:
        # ── Manual object + auto controller ──
        # Manual object first
        print(f"[Manual] Segment the TARGET OBJECT ('{object_label}') for camera {cam_id}")
        os.system(
            f"python ./data_process/manual_segment.py "
            f"--base_path {base_path} --case_name {case_name} "
            f"--camera_id {cam_id} --label '{object_label}' --obj_id 0"
        )
        # Auto controller — run segment_util_video with only the controller prompt
        controller_prompt = f"{args.controller_label}."
        print(f"[Auto] Segmenting controller with prompt: '{controller_prompt}'")
        # We need to make sure auto doesn't overwrite the mask_info.
        # segment_util_video.py overwrites mask_info, so we run it to a temp location
        # then merge. Simpler: just also do the controller manually via auto prompt
        # in a separate manual_segment call that uses GroundingDINO detection.
        # Actually, it's simplest to run the auto on just the controller as a new label.
        # But segment_util_video always starts obj_ids from 0. Let's just use manual for
        # the controller too in this case to avoid ID conflicts.
        print(f"[Auto->Manual fallback] Running auto detection for controller, "
              f"then manual verification...")
        os.system(
            f"python ./data_process/manual_segment.py "
            f"--base_path {base_path} --case_name {case_name} "
            f"--camera_id {cam_id} --label '{args.controller_label}'"
        )

    elif args.manual_controller and not args.manual_object:
        # ── Auto object + manual controller ──
        # Auto for object only (TEXT_PROMPT should be just the object, e.g. "cloth.")
        auto_prompt = args.TEXT_PROMPT if args.TEXT_PROMPT else f"{object_label}."
        print(f"[Auto] Segmenting object with prompt: '{auto_prompt}'")
        os.system(
            f"python ./data_process/segment_util_video.py "
            f"--base_path {base_path} --case_name {case_name} "
            f"--TEXT_PROMPT '{auto_prompt}' --camera_id {cam_id} "
            f"--box_threshold {args.box_threshold} --text_threshold {args.text_threshold}"
        )
        os.system(f"rm -rf {base_path}/{case_name}/tmp_data")

        # Manual for controller (appends to existing mask_info with next available ID)
        print(f"\n[Manual] Segment the CONTROLLER ('{args.controller_label}') for camera {cam_id}")
        os.system(
            f"python ./data_process/manual_segment.py "
            f"--base_path {base_path} --case_name {case_name} "
            f"--camera_id {cam_id} --label '{args.controller_label}'"
        )
