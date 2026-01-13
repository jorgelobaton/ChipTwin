# Use co-tracker to track the object and controller in the video (pick 5000 pixels in the masked area)

import torch
import imageio.v3 as iio
from utils.visualizer import Visualizer
import glob
import cv2
import numpy as np
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name

# Dynamically determine valid camera folders
camera_folders = glob.glob(f"{base_path}/{case_name}/depth/*")
valid_cameras = [f for f in camera_folders if os.path.isdir(f)]
num_cam = len(valid_cameras)

if num_cam == 0:
    raise ValueError(f"No depth folders found in {base_path}/{case_name}/depth/")

print(f"Detected {num_cam} cameras.")
device = "cuda"

# MEMORY OPTIMIZATION: Process points in batches
POINTS_PER_BATCH = 1000  # Process 1000 points at a time (adjustable)
TOTAL_POINTS = 5000      # Total points to track

def read_mask(mask_path):
    # Convert the white mask into binary mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask > 0
    return mask

def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def track_points_in_batches(video, query_pixels, device, points_per_batch=1000):
    """
    Track query points in batches to avoid OOM errors.
    Returns combined tracks and visibility for all points.
    """
    num_points = query_pixels.shape[0]
    num_batches = (num_points + points_per_batch - 1) // points_per_batch
    
    all_pred_tracks = []
    all_pred_visibility = []
    
    for batch_idx in range(num_batches):
        print(f"  Batch {batch_idx + 1}/{num_batches}")
        
        # Clear GPU cache before each batch
        torch.cuda.empty_cache()
        
        # Get batch of query points
        start_idx = batch_idx * points_per_batch
        end_idx = min(start_idx + points_per_batch, num_points)
        batch_queries = query_pixels[start_idx:end_idx]
        
        # Load cotracker for this batch
        cotracker = torch.hub.load(
            "facebookresearch/co-tracker", "cotracker3_online"
        ).to(device)
        
        # Initialize with queries
        cotracker(video_chunk=video, is_first_step=True, queries=batch_queries[None])
        
        # Process the video in chunks
        for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
            pred_tracks, pred_visibility = cotracker(
                video_chunk=video[:, ind : ind + cotracker.step * 2]
            )
        
        # Store results
        all_pred_tracks.append(pred_tracks.cpu())
        all_pred_visibility.append(pred_visibility.cpu())
        
        # Delete cotracker to free GPU memory
        del cotracker
        torch.cuda.empty_cache()
    
    # Concatenate all batches along the point dimension (N)
    combined_tracks = torch.cat(all_pred_tracks, dim=2)  # B T N 2
    combined_visibility = torch.cat(all_pred_visibility, dim=2)  # B T N 1
    
    return combined_tracks, combined_visibility

if __name__ == "__main__":
    exist_dir(f"{base_path}/{case_name}/cotracker")

    for i in range(num_cam):
        print(f"Processing {i}th camera")
        
        # Clear GPU cache before processing each camera
        torch.cuda.empty_cache()
        
        # Load the video
        frames = iio.imread(f"{base_path}/{case_name}/color/{i}.mp4", plugin="FFMPEG")
        video = (
            torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)
        )  # B T C H W
        
        # Load the first-frame mask to get all query points from all masks
        mask_paths = glob.glob(f"{base_path}/{case_name}/mask/{i}/*/0.png")
        mask = None
        for mask_path in mask_paths:
            current_mask = read_mask(mask_path)
            if mask is None:
                mask = current_mask
            else:
                mask = np.logical_or(mask, current_mask)

        # Draw the mask
        query_pixels = np.argwhere(mask)
        # Revert x and y
        query_pixels = query_pixels[:, ::-1]
        query_pixels = np.concatenate(
            [np.zeros((query_pixels.shape[0], 1)), query_pixels], axis=1
        )
        query_pixels = torch.tensor(query_pixels, dtype=torch.float32).to(device)
        
        # Randomly select TOTAL_POINTS query points
        if query_pixels.shape[0] > TOTAL_POINTS:
            indices = torch.randperm(query_pixels.shape[0])[:TOTAL_POINTS]
            query_pixels = query_pixels[indices]
        
        print(f"  Tracking {query_pixels.shape[0]} points in batches of {POINTS_PER_BATCH}")
        
        # Track points in batches
        pred_tracks, pred_visibility = track_points_in_batches(
            video, query_pixels, device, points_per_batch=POINTS_PER_BATCH
        )
        
        # Visualize results
        vis = Visualizer(
            save_dir=f"{base_path}/{case_name}/cotracker", pad_value=0, linewidth=3
        )
        # Move back to device for visualization
        vis.visualize(
            video.cpu(), 
            pred_tracks, 
            pred_visibility, 
            filename=f"{i}"
        )
        
        # Save the tracking data into npz
        track_to_save = pred_tracks[0].numpy()[:, :, ::-1]
        visibility_to_save = pred_visibility[0].numpy()
        np.savez(
            f"{base_path}/{case_name}/cotracker/{i}.npz",
            tracks=track_to_save,
            visibility=visibility_to_save,
        )
        
        # Clear GPU memory after each camera
        del video, pred_tracks, pred_visibility
        torch.cuda.empty_cache()
        
        print(f"  Camera {i} complete. Saved to {base_path}/{case_name}/cotracker/{i}.npz")
