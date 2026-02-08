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

    # 2. Setup Recording Paths
    camera_ids = sorted(cameras.keys())
    for cam_id in camera_ids:
        os.makedirs(f"{output_dir}/color/{cam_id}", exist_ok=True)
        os.makedirs(f"{output_dir}/depth/{cam_id}", exist_ok=True)

    # 3. Init Video Writers
    video_writers = {}
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    
    print("Initializing streams...")
    while True:
        d, c = cameras[camera_ids[0]].get_frame()
        if c is not None: break
        time.sleep(0.1)

    for cam_id in camera_ids:
        d, c = cameras[cam_id].get_frame() 
        if c is None: continue 
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

            # A. Synchronized Capture
            frames = {}
            for cid, cam in cameras.items():
                d, c = cam.get_frame()
                if c is not None: frames[cid] = (d, c)
            
            if len(frames) != len(cameras):
                continue

            # B. Processing / Recording
            if is_recording:
                for cid in camera_ids:
                    d, c = frames[cid]
                    depth_path = f"{output_dir}/depth/{cid}/{global_frame_count}.npy"
                    item = (depth_path, video_writers[cid], c, d)
                    save_queue.put(item)
                
                if global_frame_count % 30 == 0:
                    print(f"Recorded {global_frame_count} frames... (Q: {save_queue.qsize()})")
                global_frame_count += 1

            # C. Visualization (With Overlay)
            if not is_recording:
                display_imgs = []
                for cid in camera_ids:
                    d, c = frames[cid]
                    
                    # Depth Overlay
                    vis_scale = cameras[cid].config_data.get('visual_scale', 5)
                    d_cm = cv2.applyColorMap(cv2.convertScaleAbs(d, alpha=vis_scale/100.0), cv2.COLORMAP_JET)
                    d_cm[d==0] = 0
                    ov = cv2.addWeighted(c, 0.6, d_cm, 0.4, 0)

                    cv2.putText(ov, cid, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    display_imgs.append(cv2.resize(ov, (400, 400)))
                
                if display_imgs:
                    stack = np.hstack(display_imgs)
                    status_bar = np.zeros((60, stack.shape[1], 3), dtype=np.uint8)
                    
                    if is_timed_mode and timed_state == 1:
                        rem = max(0.0, 3.0 - (time.time() - timer_start_ts))
                        txt = f"Starting in {rem:.1f}s..."
                        col = (0, 165, 255) # Orange
                    else:
                        txt = "Standby (Space: Man, T: Timer)"
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
        print("Stopping cameras...")
        for cam in cameras.values(): cam.stop()
        
        print("Waiting for disk writes...")
        stop_saver.set()
        saver.join()
        
        for vw in video_writers.values(): vw.release()
        cv2.destroyAllWindows()

        # Update Metadata
        metadata["frame_num"] = global_frame_count
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
