import os
from argparse import ArgumentParser
import time
import logging
import json
import glob

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    default="./data/different_types",
)
parser.add_argument("--case_name", type=str, required=True)
# The category of the object used for segmentation
parser.add_argument("--category", type=str, required=True)
parser.add_argument("--controller", type=str, default="hand",
                    help="Name of the controller (e.g. 'hand', 'glove'). Used in the TEXT_PROMPT and for mask label matching.")
parser.add_argument("--box_threshold", type=float, default=0.35,
                    help="GroundingDINO box confidence threshold (lower = more permissive)")
parser.add_argument("--text_threshold", type=float, default=0.25,
                    help="GroundingDINO text confidence threshold")
parser.add_argument("--manual_controller_mask", action="store_true", default=False,
                    help="Use interactive manual segmentation for the controller instead of GroundingDINO. "
                         "Useful for robot grippers or objects that GroundingDINO cannot reliably detect.")
parser.add_argument("--manual_object_mask", action="store_true", default=False,
                    help="Use interactive manual segmentation for the target object instead of GroundingDINO.")
parser.add_argument("--shape_prior", action="store_true", default=False)
args = parser.parse_args()

# Set the debug flagspython process_data.py --base_path ./data/different_types --case_name demo_cable --category cable --controller hand --manual_segment
PROCESS_SEG = True
PROCESS_SHAPE_PRIOR = True
PROCESS_TRACK = True
PROCESS_3D = True
PROCESS_ALIGN = True
PROCESS_FINAL = True

base_path = args.base_path
case_name = args.case_name
category = args.category
CONTROLLER_NAME = args.controller
MANUAL_CONTROLLER_MASK = args.manual_controller_mask
MANUAL_OBJECT_MASK = args.manual_object_mask
ANY_MANUAL = MANUAL_CONTROLLER_MASK or MANUAL_OBJECT_MASK

# Build TEXT_PROMPT based on what's being auto-segmented
if not ANY_MANUAL:
    TEXT_PROMPT = f"{category}.{CONTROLLER_NAME}"
elif MANUAL_CONTROLLER_MASK and not MANUAL_OBJECT_MASK:
    TEXT_PROMPT = f"{category}."   # only object in auto prompt
elif MANUAL_OBJECT_MASK and not MANUAL_CONTROLLER_MASK:
    TEXT_PROMPT = f"{CONTROLLER_NAME}."  # only controller in auto prompt
else:
    TEXT_PROMPT = ""  # both manual — no auto prompt needed

SHAPE_PRIOR = args.shape_prior

logger = None


def setup_logger(log_file="timer.log"):
    global logger 

    if logger is None:
        logger = logging.getLogger("GlobalLogger")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter("%(message)s"))

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)


setup_logger()


def existDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


class Timer:
    def __init__(self, task_name):
        self.task_name = task_name

    def __enter__(self):
        self.start_time = time.time()
        logger.info(
            f"!!!!!!!!!!!! {self.task_name}: Processing {case_name} !!!!!!!!!!!!"
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.time() - self.start_time
        logger.info(
            f"!!!!!!!!!!! Time for {self.task_name}: {elapsed_time:.2f} sec !!!!!!!!!!!!"
        )


if PROCESS_SEG:
    # Get the masks of the controller and the object
    if ANY_MANUAL:
        # At least one of controller/object uses manual interactive segmentation
        manual_flags = ""
        if MANUAL_CONTROLLER_MASK:
            manual_flags += " --manual_controller"
        if MANUAL_OBJECT_MASK:
            manual_flags += " --manual_object"

        timer_label = "Video Segmentation (manual"
        if MANUAL_CONTROLLER_MASK and MANUAL_OBJECT_MASK:
            timer_label += " controller+object)"
        elif MANUAL_CONTROLLER_MASK:
            timer_label += " controller)"
        else:
            timer_label += " object)"

        with Timer(timer_label):
            os.system(
                f"python ./data_process/manual_segment_video.py --base_path {base_path} --case_name {case_name} "
                f"--TEXT_PROMPT '{TEXT_PROMPT}'{manual_flags} "
                f"--controller_label '{CONTROLLER_NAME}' --object_label '{category}' "
                f"--box_threshold {args.box_threshold} --text_threshold {args.text_threshold}"
            )
    else:
        # Fully automatic segmentation using GroundedSAM2
        with Timer("Video Segmentation"):
            os.system(
                f"python ./data_process/segment.py --base_path {base_path} --case_name {case_name} --TEXT_PROMPT '{TEXT_PROMPT}' --box_threshold {args.box_threshold} --text_threshold {args.text_threshold}"
            )


if PROCESS_SHAPE_PRIOR and SHAPE_PRIOR:
    # Shape prior from point cloud (replaces Trellis + alignment)
    # NOTE: This runs after PROCESS_3D because it needs track_process_data.pkl.
    #       The old Trellis pipeline (image_upscale -> segment_util_image -> shape_prior.py -> align.py)
    #       is no longer needed. This directly produces shape/matching/final_mesh.glb from the
    #       observed shell points, which data_process_sample.py consumes.
    pass  # Actual execution deferred until after PROCESS_3D (see below)

if PROCESS_TRACK:
    # Get the dense tracking of the object using Co-tracker
    with Timer("Dense Tracking"):
        os.system(
            f"python ./data_process/dense_track.py --base_path {base_path} --case_name {case_name}"
        )

if PROCESS_3D:
    # Get the pcd in the world coordinate from the raw observations
    with Timer("Lift to 3D"):
        os.system(
            f"python ./data_process/data_process_pcd.py --base_path {base_path} --case_name {case_name}"
        )

    # Further process and filter the noise of object and controller masks
    with Timer("Mask Post-Processing"):
        os.system(
            f"python ./data_process/data_process_mask.py --base_path {base_path} --case_name {case_name} --controller_name {CONTROLLER_NAME}"
        )

    # Process the data tracking
    with Timer("Data Tracking"):
        os.system(
            f"python ./data_process/data_process_track.py --base_path {base_path} --case_name {case_name}"
        )

if PROCESS_ALIGN and SHAPE_PRIOR:
    # Generate shape prior from observed point cloud shell (replaces Trellis + alignment)
    # Creates a watertight alpha-shape mesh from the tracked object points
    with Timer("Shape Prior from PCD"):
        os.system(
        f"python ./data_process/shape_prior_pcd.py --base_path {base_path} --case_name {case_name}"
        )

if PROCESS_FINAL:
    # Get the final PCD used for the inverse physics with/without the shape prior
    with Timer("Final Data Generation"):
        if SHAPE_PRIOR:
            os.system(
                f"python ./data_process/data_process_sample.py --base_path {base_path} --case_name {case_name} --shape_prior"
            )
        else:
            os.system(
                f"python ./data_process/data_process_sample.py --base_path {base_path} --case_name {case_name}"
            )

    # Save the train test split
    frame_len = len(glob.glob(f"{base_path}/{case_name}/pcd/*.npz"))
    split = {}
    split["frame_len"] = frame_len
    split["train"] = [0, int(frame_len * 0.7)]
    split["test"] = [int(frame_len * 0.7), frame_len]
    with open(f"{base_path}/{case_name}/split.json", "w") as f:
        json.dump(split, f)
