import pyrealsense2 as rs
try:
    from pyk4a import connected_device_count
    from cams.camera_kinect import KinectCamera
    HAS_KINECT = True
except ImportError:
    HAS_KINECT = False
    def connected_device_count(): return 0
    KinectCamera = None

import time
import os
import cv2
import numpy as np
import json
from argparse import ArgumentParser
import queue
import threading
import shutil

# Import Local Camera Classes
from cams.camera_d405 import D405Camera
# from camera_kinect import KinectCamera  # Moved to try/except block above

def saver_thread(q, stop_event):
    """
    Worker thread to save frames.
    Saves Depth as NPY and Color as MP4 frames to avoid I/O bottlenecks.
    """
    while not stop_event.is_set() or not q.empty():
        try:
            item = q.get(timeout=0.5)
        except queue.Empty:
            continue

        (depth_path, video_writer, video_frame, depth_data) = item
        
        # Write Video
        if video_writer:
            video_writer.write(video_frame)
        
        # Write Depth
        np.save(depth_path, depth_data)
        
        q.task_done()


def camera_capture_thread(cam, latest_frame, lock, stop_event):
    """
    Per-camera thread: continuously calls get_frame() and stores the latest result.
    This decouples heavy per-frame processing (filters, rotation, crop) from the
    main loop, so the main loop never blocks waiting for a single camera.
    """
    while not stop_event.is_set():
        result = cam.get_frame()
        if result[1] is not None:
            with lock:
                latest_frame[0] = result

def discover_cameras():
    print("--- Discovery Mode ---")
    ctx = rs.context()
    d405_serials = []
    
    for dev in ctx.query_devices():
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        if "D405" in name:
            d405_serials.append(serial)
            
    k_count = connected_device_count()
    k_indices = list(range(k_count))
    
    return d405_serials, k_indices

def main(args):
    # Output Setup
    output_dir = os.path.join("./data/different_types", args.output_dir)
    output_dir = os.path.abspath(output_dir)
    if os.path.exists(output_dir):
        print(f"[INFO] Existing directory found at {output_dir}. Deleting...")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Saver Thread
    save_queue = queue.Queue()
    stop_saver = threading.Event()
    saver = threading.Thread(target=saver_thread, args=(save_queue, stop_saver), daemon=True)
    saver.start()

    # Load Camera Map if exists
    camera_map = {}
    if os.path.exists("cams/camera_map.json"):
        try:
            with open("cams/camera_map.json", 'r') as f:
                camera_map = json.load(f)
            print(f"[INFO] Loaded camera map with {len(camera_map)} entries.")
        except Exception as e:
            print(f"[WARN] Failed to load camera_map.json: {e}")

    # 1. Initialize Cameras
    cameras = {}
    d405_serials, k_indices = discover_cameras()
    
    # Init D405s
    # Sort serials to ensure deterministic config assignment
    d405_serials.sort()
    
    for i, serial in enumerate(d405_serials):
        try:
            cid = f"D405_{serial}"
            
            # Priority: Map -> Index -> Default
            if serial in camera_map:
                conf = camera_map[serial]
                # If not found at provided path, check under cams/
                if not os.path.exists(conf) and os.path.exists(os.path.join("cams", conf)):
                    conf = os.path.join("cams", conf)
            elif i == 0: conf = "cams/config_d405_0.json"
            elif i == 1: conf = "cams/config_d405_1.json"
            else: conf = "cams/config_d405.json"
            
            print(f"[INIT] {cid} using {conf}")
            cameras[cid] = D405Camera(serial_number=serial, config_file=conf)
        except Exception as e: print(f"[ERR] {cid}: {e}")

    # Init Kinects
    for idx in k_indices:
        try:
            cid = f"Kinect_{idx}"
            print(f"[INIT] {cid}")
            cameras[cid] = KinectCamera(device_index=idx)
        except Exception as e: print(f"[ERR] {cid}: {e}")

    if not cameras:
        print("[FAIL] No cameras found.")
        return

    # 1b. Start per-camera capture threads
    # Each thread runs get_frame() continuously so the main loop never blocks.
    cam_latest = {}   # cid -> [latest (depth, color)] (list so we can mutate in-place)
    cam_locks  = {}   # cid -> threading.Lock
    stop_capture = threading.Event()
    capture_threads = []
    for cid, cam in cameras.items():
        latest = [None]
        lock = threading.Lock()
        cam_latest[cid] = latest
        cam_locks[cid]  = lock
        t = threading.Thread(target=camera_capture_thread,
                             args=(cam, latest, lock, stop_capture),
                             daemon=True)
        t.start()
        capture_threads.append(t)

    # 2. Setup Recording Paths
    camera_ids = sorted(cameras.keys())
    for cam_id in camera_ids:
        os.makedirs(f"{output_dir}/color/{cam_id}", exist_ok=True)
        os.makedirs(f"{output_dir}/depth/{cam_id}", exist_ok=True)

    # 3. Init Video Writers
    video_writers = {}
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    print("Initializing streams...")
    # Wait until all cameras have at least one frame in their buffer
    for cid in camera_ids:
        while True:
            with cam_locks[cid]:
                frame = cam_latest[cid][0]
            if frame is not None and frame[1] is not None:
                break
            time.sleep(0.05)

    for cam_id in camera_ids:
        with cam_locks[cam_id]:
            frame = cam_latest[cam_id][0]
        if frame is None or frame[1] is None:
            continue
        c = frame[1]
        h, w = c.shape[:2]
        v_path = f"{output_dir}/color/{cam_id}.mp4"
        video_writers[cam_id] = cv2.VideoWriter(v_path, fourcc, 30.0, (w, h))

    # 4. Extract Intrinsics
    print("Extracting Metadata...")
    intrinsics_ordered = []
    depth_scales_ordered = []
    
    for cam_id in camera_ids:
        cam = cameras[cam_id]
        K = cam.get_intrinsics_matrix()
        intrinsics_ordered.append(K.tolist())
        depth_scales_ordered.append(cam.scale)

    metadata = {
        "camera_ids": camera_ids,
        "intrinsics": intrinsics_ordered,
        "depth_scales": depth_scales_ordered,
        "fps": 30,
        "WH": [720, 720],
        "recording_log": {}
    }

    print("\n--- READY TO RECORD ---")
    print("[SPACE] Toggle Record | [Q] Quit | [T] Timer (3s wait -> 10s rec)")
    
    cv2.namedWindow('Recorder', cv2.WINDOW_AUTOSIZE)
    is_recording = False
    global_frame_count = 0
    
    # Timer State
    is_timed_mode = False
    timed_state = 0 # 0: Idle, 1: Countdown, 2: Recording
    timer_start_ts = 0.0

    # FPS tracking
    fps_window = 60  # rolling window size (frames)
    frame_timestamps = []   # timestamps of captured main-loop iterations
    record_timestamps = []  # timestamps of recorded frames (for recording FPS)
    display_fps = 0.0
    record_fps  = 0.0

    try:
        while True:
            # Timer Logic
            if is_timed_mode:
                now = time.time()
                if timed_state == 1: # Countdown
                    if now - timer_start_ts >= 3.0:
                        timed_state = 2
                        is_recording = True
                        timer_start_ts = now # Reset for duration
                        print(">> TIMER: START RECORDING (8s)")
                elif timed_state == 2: # Timed Recording
                    if now - timer_start_ts >= 10.0:
                        is_recording = False
                        is_timed_mode = False
                        timed_state = 0
                        print(">> TIMER: STOP RECORDING")

            # A. Synchronized Capture — read latest frame from each camera thread
            frames = {}
            for cid in camera_ids:
                with cam_locks[cid]:
                    frame = cam_latest[cid][0]
                if frame is not None and frame[1] is not None:
                    frames[cid] = frame
            
            if len(frames) != len(cameras):
                continue

            # Track main-loop FPS (rolling window)
            now = time.monotonic()
            frame_timestamps.append(now)
            if len(frame_timestamps) > fps_window:
                frame_timestamps.pop(0)
            if len(frame_timestamps) >= 2:
                elapsed = frame_timestamps[-1] - frame_timestamps[0]
                display_fps = (len(frame_timestamps) - 1) / elapsed if elapsed > 0 else 0.0

            # B. Processing / Recording
            if is_recording:
                rec_now = time.monotonic()
                record_timestamps.append(rec_now)
                if len(record_timestamps) > fps_window:
                    record_timestamps.pop(0)
                if len(record_timestamps) >= 2:
                    elapsed = record_timestamps[-1] - record_timestamps[0]
                    record_fps = (len(record_timestamps) - 1) / elapsed if elapsed > 0 else 0.0

                for cid in camera_ids:
                    d, c = frames[cid]
                    depth_path = f"{output_dir}/depth/{cid}/{global_frame_count}.npy"
                    item = (depth_path, video_writers[cid], c, d)
                    save_queue.put(item)
                
                if global_frame_count % 30 == 0:
                    print(f"Recorded {global_frame_count} frames... (Q: {save_queue.qsize()}) | FPS: {record_fps:.1f}")
                global_frame_count += 1

            # C. Visualization (With Overlay)
            display_imgs = []
            for cid in camera_ids:
                d, c = frames[cid]
                
                if not is_recording:
                    # Depth Overlay
                    vis_scale = cameras[cid].config_data.get('visual_scale', 5)
                    d_cm = cv2.applyColorMap(cv2.convertScaleAbs(d, alpha=vis_scale/100.0), cv2.COLORMAP_JET)
                    d_cm[d==0] = 0
                    ov = cv2.addWeighted(c, 0.6, d_cm, 0.4, 0)
                else:
                    ov = c.copy()

                cv2.putText(ov, cid, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                display_imgs.append(cv2.resize(ov, (400, 400)))
            
            if display_imgs:
                stack = np.hstack(display_imgs)
                status_bar = np.zeros((60, stack.shape[1], 3), dtype=np.uint8)
                
                if is_recording:
                    txt = f"REC  frame:{global_frame_count}  FPS:{record_fps:.1f}"
                    col = (0, 0, 255) # Red while recording
                elif is_timed_mode and timed_state == 1:
                    rem = max(0.0, 3.0 - (time.time() - timer_start_ts))
                    txt = f"Starting in {rem:.1f}s...  preview FPS:{display_fps:.1f}"
                    col = (0, 165, 255) # Orange
                else:
                    txt = f"Standby (Space: Man, T: Timer)  FPS:{display_fps:.1f}"
                    col = (0, 255, 0)

                cv2.putText(status_bar, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                final_vis = np.vstack((stack, status_bar))
                cv2.imshow("Recorder", final_vis)

            # D. Input
            k = cv2.waitKey(1)
            if k in [ord('q'), 27]: break
            elif k == ord(' ') and not is_timed_mode:
                is_recording = not is_recording
                if is_recording:
                    print(f">> START RECORDING")
                else:
                    print(f">> STOP RECORDING")
            elif k == ord('t'):
                if not is_recording and not is_timed_mode:
                    is_timed_mode = True
                    timed_state = 1
                    timer_start_ts = time.time()
                    print(f">> TIMER STARTED (3s Countdown)")

    finally:
        print("Stopping capture threads...")
        stop_capture.set()
        for t in capture_threads:
            t.join(timeout=2.0)

        print("Stopping cameras...")
        for cam in cameras.values(): cam.stop()
        
        print("Waiting for disk writes...")
        stop_saver.set()
        saver.join()
        
        for vw in video_writers.values(): vw.release()
        cv2.destroyAllWindows()

        # Print final FPS summary
        if record_timestamps and len(record_timestamps) >= 2:
            total_elapsed = record_timestamps[-1] - record_timestamps[0]
            final_fps = (len(record_timestamps) - 1) / total_elapsed if total_elapsed > 0 else 0.0
            print(f"\n[FPS] Recorded {global_frame_count} frames  |  avg record FPS: {final_fps:.2f}")
        elif global_frame_count > 0:
            print(f"\n[FPS] Recorded {global_frame_count} frames (not enough data for FPS calc)")

        # Update Metadata
        real_fps = 0.0
        if record_timestamps and len(record_timestamps) >= 2:
            total_elapsed = record_timestamps[-1] - record_timestamps[0]
            real_fps = (len(record_timestamps) - 1) / total_elapsed if total_elapsed > 0 else 0.0
        metadata["frame_num"] = global_frame_count
        metadata["real_fps"] = round(real_fps, 2)
        with open(f"{output_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        # Copy Calibration
        if os.path.exists("calibrate.pkl"):
            shutil.copy("calibrate.pkl", f"{output_dir}/calibrate.pkl")
            print("Copied global calibration file.")

        # ==========================================
        # POST-PROCESS: Extract PNGs from MP4
        # ==========================================
        if global_frame_count > 0:
            print("\n[POST-PROCESS] Extracting synchronized PNGs from video...")
            for cam_id in camera_ids:
                vid_path = f"{output_dir}/color/{cam_id}.mp4"
                out_path = f"{output_dir}/color/{cam_id}" # Target folder
                
                if not os.path.exists(vid_path): continue
                
                cap = cv2.VideoCapture(vid_path)
                cnt = 0
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Save as 0.png, 1.png matching the NPY indices
                    cv2.imwrite(f"{out_path}/{cnt}.png", frame)
                    cnt += 1
                cap.release()
                print(f"  Camera {cam_id}: Extracted {cnt} PNGs.")

        print(f"Saved dataset to {output_dir}")

        if args.auto_pcd:
            print("Generating PCDs...")
            # subprocess.run(...) logic here
            pass 

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/test_capture")
    parser.add_argument("--auto_pcd", action="store_true")
    args = parser.parse_args()
    main(args)
