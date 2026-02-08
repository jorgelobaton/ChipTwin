import ffmpeg
import subprocess
import sys
import os
import glob
import json
from argparse import ArgumentParser

def get_fps_and_frames(path):
    # Use ffprobe to get FPS and total frame count
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,nb_frames",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd).decode().strip().split("\n")
    if len(out) < 2:
        raise RuntimeError("Could not read fps/nb_frames from video.")
    
    fps_num, fps_den = map(int, out[0].split("/"))
    fps = fps_num / fps_den
    # Some files don't report nb_frames in metadata; fallback to 0 if missing
    total_frames = int(out[1]) if out[1].isdigit() else 0
    return fps, total_frames

def trim_last_n_frames(input_path, output_path, n_frames=8):
    fps, total_frames = get_fps_and_frames(input_path)
    
    # If total_frames is unknown, we calculate from duration
    if total_frames == 0:
        duration = float(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", input_path
        ]).decode().strip())
        total_frames = int(duration * fps)

    if total_frames <= n_frames:
        raise ValueError(f"Video ({total_frames} frames) is too short to trim {n_frames} frames.")
    # Calculate end time: (Total Frames - n_frames) / FPS
    end_time = (total_frames - n_frames) / fps

    # Apply re-encoding (cannot use 'copy' with 'trim' filter)
    try:
        (
            ffmpeg
            .input(input_path)
            .trim(start=0, end=end_time)
            .setpts("PTS-STARTPTS")
            .output(output_path, vcodec="libx264", acodec="aac", crf=23)
            .run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        )
        print(f"Successfully saved trimmed video to: {output_path}")
    except ffmpeg.Error as e:
        print(e.stderr.decode(), file=sys.stderr)
        sys.exit(1)

def trim_experiment(demo_path, n_frames):
    """
    Finds videos starting with D405 in color/ folder of the demo,
    trims them, and removes the corresponding last N frames from 
    color and depth image/data folders.
    """
    color_dir = os.path.join(demo_path, "color")
    depth_dir = os.path.join(demo_path, "depth")

    if not os.path.exists(color_dir):
        print(f"[ERR] Color directory not found: {color_dir}")
        return

    # Find D405 videos in color/
    videos = glob.glob(os.path.join(color_dir, "D405_*.mp4")) + \
             glob.glob(os.path.join(color_dir, "d405_*.mp4"))
    
    # Filter out duplicates if any
    videos = list(set(videos))

    if not videos:
        print(f"[INFO] No D405 videos found in {color_dir}")
        return

    for v_path in videos:
        v_name = os.path.basename(v_path)
        serial = v_name.replace(".mp4", "")
        print(f"\n[TRIM] Processing camera: {serial}")

        # 1. Trim the Video
        # We trim in-place by creating a temp file and replacing
        temp_v = v_path.replace(".mp4", "_trimmed.mp4")
        try:
            trim_last_n_frames(v_path, temp_v, n_frames)
            os.remove(v_path)
            os.rename(temp_v, v_path)
        except Exception as e:
            print(f"[ERR] Failed to trim video {v_name}: {e}")
            if os.path.exists(temp_v): os.remove(temp_v)
            continue

        # 2. Trim Color Folder
        c_folder = os.path.join(color_dir, serial)
        if os.path.exists(c_folder):
            files = sorted(glob.glob(os.path.join(c_folder, "*")), key=lambda x: int(os.path.basename(x).split(".")[0]) if os.path.basename(x).split(".")[0].isdigit() else x)
            if len(files) >= n_frames:
                to_delete = files[-n_frames:]
                for f in to_delete:
                    os.remove(f)
                print(f"[INFO] Removed {n_frames} frames from {c_folder}")
            else:
                print(f"[WARN] Folder {c_folder} has fewer than {n_frames} frames.")

        # 3. Trim Depth Folder
        d_folder = os.path.join(depth_dir, serial)
        if os.path.exists(d_folder):
            files = sorted(glob.glob(os.path.join(d_folder, "*")), key=lambda x: int(os.path.basename(x).split(".")[0]) if os.path.basename(x).split(".")[0].isdigit() else x)
            if len(files) >= n_frames:
                to_delete = files[-n_frames:]
                for f in to_delete:
                    os.remove(f)
                print(f"[INFO] Removed {n_frames} frames from {d_folder}")
            else:
                print(f"[WARN] Folder {d_folder} has fewer than {n_frames} frames.")

    # 4. Update metadata.json
    metadata_path = os.path.join(demo_path, "metadata.json")
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
            
            if "frame_num" in meta:
                old_num = meta["frame_num"]
                new_num = max(0, old_num - n_frames)
                meta["frame_num"] = new_num
                with open(metadata_path, 'w') as f:
                    json.dump(meta, f, indent=4)
                print(f"[INFO] Updated metadata.json: frame_num {old_num} -> {new_num}")
            else:
                print(f"[WARN] 'frame_num' not found in metadata.json")
        except Exception as e:
            print(f"[ERR] Failed to update metadata.json: {e}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Trim last N frames from D405 videos and data folders.")
    parser.add_argument("--demo_path", type=str, help="Path to the experiment/demo folder.")
    parser.add_argument("--category", type=str, help="Category name under data/different_types.")
    parser.add_argument("--demo", type=str, help="Demo name.")
    parser.add_argument("--n_frames", type=int, default=8, help="Number of frames to trim from the end.")
    
    # Legacy support
    parser.add_argument("input", nargs="?", help="Input video file (single file mode)")
    parser.add_argument("output", nargs="?", help="Output video file (single file mode)")
    parser.add_argument("frames", nargs="?", type=int, help="Frames to trim (single file mode)")

    args = parser.parse_args()

    if args.demo_path or (args.category and args.demo):
        if args.demo_path:
            path = args.demo_path
        else:
            path = os.path.join("data/different_types", args.category, args.demo)
        
        path = os.path.abspath(path)
        print(f"[START] Trimming experiment at {path} by {args.n_frames} frames.")
        trim_experiment(path, args.n_frames)
    elif args.input and args.output and args.frames is not None:
        trim_last_n_frames(args.input, args.output, args.frames)
    else:
        parser.print_help()
        sys.exit(1)
