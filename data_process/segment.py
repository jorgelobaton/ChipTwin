# Process to get the masks of the controller and the object
import os
import glob
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--TEXT_PROMPT", type=str, required=True)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
TEXT_PROMPT = args.TEXT_PROMPT
# List all subdirectories in depth/ to get camera IDs (names)
camera_ids = sorted([os.path.basename(p) for p in glob.glob(f"{base_path}/{case_name}/depth/*")])
print(f"Processing {case_name}")

for cam_id in camera_ids:
    print(f"Processing {case_name} camera {cam_id}")
    os.system(
        f"python ./data_process/segment_util_video.py --base_path {base_path} --case_name {case_name} --TEXT_PROMPT '{TEXT_PROMPT}' --camera_id {cam_id}"
    )
    os.system(f"rm -rf {base_path}/{case_name}/tmp_data")
