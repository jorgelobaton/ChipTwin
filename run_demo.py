import pyrealsense2 as rs
from pyk4a import PyK4A, connected_device_count
import time
import os
import cv2
import numpy as np
import json
import pickle
from argparse import ArgumentParser
import queue
import threading
import shutil
import glob

# Import your CameraSystem
try:
    from qqtt.env.camera.camera_system import CameraSystem
except ImportError:
    try:
        from camera_system import CameraSystem
    except ImportError:
        print("Error: Could not import CameraSystem.")
        exit(1)

def saver_thread(q, stop_event):
    """Worker thread to save frames from a queue to disk."""
    while not stop_event.is_set() or not q.empty():
        try:
            # Block for up to 1 second, then check stop_event
            item = q.get(timeout=1)
        except queue.Empty:
            continue

        # Unpack item and save
        # Note: We do NOT save individual PNGs here to maintain high FPS.
        # We save Video (MP4) + Depth (NPY). Frames are extracted from MP4 later.
        (depth_path, video_writer, color_data, depth_data) = item
        
        if video_writer:
            video_writer.write(color_data)
        np.save(depth_path, depth_data)
        q.task_done()

def discover_cameras():
    print("--- Discovery Mode ---")
    realsense_ctx = rs.context()
    rs_devices = realsense_ctx.query_devices()
    
    standard_serials = []
    d405_serials = []
    
    for dev in rs_devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"Found RealSense: {name} (S/N: {serial})")
        if "D405" in name:
            d405_serials.append(serial)
        else:
            standard_serials.append(serial)
            
    kinect_count = connected_device_count()
    kinect_indices = list(range(kinect_count))
    if kinect_count > 0:
        print(f"Found {kinect_count} Azure Kinect(s)")
    
    return standard_serials, d405_serials, kinect_indices

def main(output_dir):
    output_dir = "./data/different_types/" + output_dir.strip('/')

    # Start the saver thread
    save_queue = queue.Queue()
    stop_saver = threading.Event()
    saver = threading.Thread(target=saver_thread, args=(save_queue, stop_saver))
    saver.start()

    rs_serials, d405_serials, k_indices = discover_cameras()
    
    if not (rs_serials or d405_serials or k_indices):
        print("\nNo cameras detected! Please check USB connections.")
        stop_saver.set()
        saver.join()
        return

    # Configuration
    rs_config = {"WH": [848, 480], "fps": 30, "serials": rs_serials}
    d405_config = {"WH": [1280, 720], "fps": 30, "serials": d405_serials}
    kinect_config = {"mode": "WFOV_BINNED", "fps": 30, "indices": k_indices, "WH": [1280, 720]}

    print("\n--- Initializing System ---")
    cam_sys = CameraSystem(
        realsense_config=rs_config,
        d405_config=d405_config,
        kinect_config=kinect_config,
        exposure=10000, 
        gain=60
    )
    
    # Get first observation to determine camera mapping
    print("Waiting for camera data...")
    first_obs = None
    while first_obs is None:
        first_obs = cam_sys.get_observation()
        if first_obs:
            break
        time.sleep(0.1)
    
    # Create INTEGER camera ID mapping (PhysTwin requirement)
    # Map: internal_key -> PhysTwin_integer_id
    camera_keys = sorted(first_obs.keys(), key=lambda x: (isinstance(x, str), x))
    camera_mapping = {key: idx for idx, key in enumerate(camera_keys)}
    
    print(f"\nCamera Mapping (internal -> PhysTwin ID):")
    for internal_key, pt_id in camera_mapping.items():
        print(f"  {internal_key} -> {pt_id}")
    
    # Initialize folders
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)
    
    for pt_id in camera_mapping.values():
        os.makedirs(f"{output_dir}/color/{pt_id}", exist_ok=True)
        os.makedirs(f"{output_dir}/depth/{pt_id}", exist_ok=True)

    # Initialize Video Writers
    video_writers = {}
    # Use avc1 (H.264) for better compatibility with standard players
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    for internal_key, pt_id in camera_mapping.items():
        # Get resolution from first_obs
        h, w = first_obs[internal_key]["color"].shape[:2]
        video_path = f"{output_dir}/color/{pt_id}.mp4"
        video_writers[pt_id] = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
        print(f"Initialized video recording for Camera {pt_id} at {w}x{h} to {video_path}")
    
    # Metadata structure matching PhysTwin format
    metadata = {
        "recording": {str(pt_id): {} for pt_id in camera_mapping.values()},
        "intrinsics": [],
        "serial_numbers": [],
        "fps": 30,
        "WH": [1280, 720]  # Will be overridden per camera if needed
    }
    
    # Calculate time offsets for hardware synchronization analysis
    print("\nSynchronizing clocks...")
    time_offsets = {}
    ref_ts = None
    for i in range(10): # Average over 10 frames
        obs = cam_sys.get_observation()
        if not obs: continue
        for key in camera_keys:
            if ref_ts is None:
                ref_ts = obs[key]["timestamp"]
                time_offsets[key] = 0
            elif key not in time_offsets:
                time_offsets[key] = obs[key]["timestamp"] - ref_ts
    
    # Extract intrinsics in PhysTwin format (list of 3x3 matrices)
    intrinsics_list = []
    for internal_key in camera_keys:
        pt_id = camera_mapping[internal_key]
        
        # Find corresponding device
        found = False
        for dev in cam_sys.devices:
            dev_ints = dev.get_intrinsics()
            
            if hasattr(dev, 'device_index'):  # Kinect
                if f"kinect_{dev.device_index}" == internal_key:
                    if len(dev_ints) > 0:
                        # dev_ints[0] is (matrix, distortion) for Kinect
                        intrinsics_list.append(dev_ints[0][0].tolist())
                        found = True
                        break
            else:  # RealSense
                # For RealSense, dev_ints is a list
                # Match by checking if internal_key is an integer in range
                if isinstance(internal_key, int):
                    # Assuming MultiRealsense returns intrinsics in order
                    if internal_key < len(dev_ints):
                        mat = dev_ints[internal_key]
                        if hasattr(mat, "fx"):  # PyRealsense2 Intrinsics object
                            mat_arr = [
                                [mat.fx, 0, mat.ppx],
                                [0, mat.fy, mat.ppy],
                                [0, 0, 1]
                            ]
                            intrinsics_list.append(mat_arr)
                        elif hasattr(mat, "tolist"):
                            intrinsics_list.append(mat.tolist())
                        found = True
                        break
        
        if not found:
            print(f"Warning: Could not find intrinsics for {internal_key}, using identity")
            intrinsics_list.append([[1, 0, 640], [0, 1, 360], [0, 0, 1]])
    
    metadata["intrinsics"] = intrinsics_list
    
    # Recording state
    last_timestamps = {key: -1 for key in camera_keys}
    is_recording_active = False
    global_frame_count = 0
    last_print_time = time.time()

    print(f"\nSystem Ready.")
    print(f"Press 'Space' to START/PAUSE recording to '{output_dir}/'")
    print(f"Press 'Esc' to exit and save metadata.")
    print(f"Recording will save as: color/0/, color/1/, etc.")

    try:
        while True:
            obs = cam_sys.get_observation()
            if not obs: continue
            
            # 1. PREVIEW LOOP
            # Skip preview entirely during recording to maximize FPS
            if not cam_sys.recording:
                for internal_key, data in obs.items():
                    pt_id = camera_mapping[internal_key]
                    img = data['color'].copy()
                    
                    # Resize for display if huge
                    if img.shape[0] > 720:
                        scale = 720 / img.shape[0]
                        img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))
                    
                    # Show PhysTwin ID in window
                    status = "REC" if cam_sys.recording else "IDLE"
                    cv2.putText(img, f"PT_ID: {pt_id} [{status}]", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    window_name = f"Camera {pt_id}"
                    cv2.imshow(window_name, img)
            else:
                # Still need to handle GUI events to detect keypresses (Space/Esc)
                cv2.waitKey(1)

            # 2. RECORDING LOGIC
            if cam_sys.recording:
                if not is_recording_active:
                    print(f">> Recording STARTED (Frame {global_frame_count})")
                    is_recording_active = True
                
                # Check if we have new data from ANY camera to trigger a sync set
                any_new_data = False
                for key in camera_keys:
                    if obs[key]["timestamp"] != last_timestamps[key]:
                        any_new_data = True
                        break
                
                # Save synchronized frame set
                if any_new_data:
                    for internal_key in camera_keys:
                        pt_id = camera_mapping[internal_key]
                        data = obs[internal_key]
                        ts = data["timestamp"]
                        last_timestamps[internal_key] = ts
                        
                        # --- Queue data for saving instead of writing directly ---
                        # We no longer save individual PNGs here. Frames are extracted from MP4 later.
                        # color_path = f"{output_dir}/color/{pt_id}/{global_frame_count}.png" 
                        depth_path = f"{output_dir}/depth/{pt_id}/{global_frame_count}.npy"
                        writer = video_writers.get(pt_id)
                        
                        # Pack 4 items: (depth_path, writer, color_data, depth_data)
                        item_to_save = (depth_path, writer, data["color"], data["depth"])
                        save_queue.put(item_to_save)
                        
                        # Log both hardware and host receive timestamps for robust analysis
                        hw_ts = ts
                        recv_ts = data.get('camera_receive_timestamp', time.time())
                        # store as a small dict so we can analyze both hardware and host timings
                        metadata["recording"][str(pt_id)][str(global_frame_count)] = {
                            'hw_ts': float(hw_ts),
                            'recv_ts': float(recv_ts)
                        }
                    
                    global_frame_count += 1
                    
                    # Print progress and queue size
                    if time.time() - last_print_time > 1.0:
                        print(f"  Recorded {global_frame_count} frames... (Save queue size: {save_queue.qsize()})")
                        last_print_time = time.time()
            
            elif is_recording_active:
                print(f">> Recording PAUSED at frame {global_frame_count}")
                is_recording_active = False

            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            if cam_sys.end:
                break

    except KeyboardInterrupt:
        pass
    finally:
        print(f"\nStopping... Recorded {global_frame_count} frames total.")
        
        # Wait for saver thread to finish
        print("Waiting for all frames to be saved...")
        stop_saver.set()
        saver.join()
        print("Done saving.")

        # --- Post-Recording Analysis ---
        if global_frame_count > 1:
            print("\n--- Post-Recording Analysis ---")
            
            # 1. Calculate Actual Frame Rate using host receive timestamps (more reliable)
            recording_data = metadata['recording']
            cam_ids = list(recording_data.keys())
            ref_cam_id = cam_ids[0]

            # Ensure frame keys exist before accessing
            if '0' in recording_data[ref_cam_id] and str(global_frame_count - 1) in recording_data[ref_cam_id]:
                first_ts = recording_data[ref_cam_id]['0']['recv_ts']
                last_ts = recording_data[ref_cam_id][str(global_frame_count - 1)]['recv_ts']
                total_duration = last_ts - first_ts

                if total_duration > 0:
                    actual_fps = (global_frame_count - 1) / total_duration
                    print(f"  - Actual Average Frame Rate: {actual_fps:.2f} FPS")
                else:
                    print("  - Could not calculate frame rate (duration is zero).")
            else:
                print("  - Could not calculate frame rate (missing start/end timestamps).")

            # 2. Check Time Synchronization (report both hardware and host-based values)
            hw_sync_errors = []
            recv_sync_errors = []
            for frame_idx in range(global_frame_count):
                hw_ts_for_frame = []
                recv_ts_for_frame = []
                for cam_id in cam_ids:
                    rec = recording_data[cam_id].get(str(frame_idx))
                    if rec is not None:
                        # rec is a dict with 'hw_ts' and 'recv_ts'
                        hw_ts_for_frame.append(rec['hw_ts'])
                        recv_ts_for_frame.append(rec['recv_ts'])

                if len(hw_ts_for_frame) > 1:
                    hw_sync_errors.append((max(hw_ts_for_frame) - min(hw_ts_for_frame)) * 1000)
                if len(recv_ts_for_frame) > 1:
                    recv_sync_errors.append((max(recv_ts_for_frame) - min(recv_ts_for_frame)) * 1000)

            if recv_sync_errors:
                avg_recv = np.mean(recv_sync_errors)
                max_recv = np.max(recv_sync_errors)
                std_recv = np.std(recv_sync_errors)
                print("  - Host (receive) Synchronization Quality:")
                print(f"    - Average Sync Error: {avg_recv:.2f} ms")
                print(f"    - Max Sync Error: {max_recv:.2f} ms")
                print(f"    - Std Dev of Error: {std_recv:.2f} ms")

            if hw_sync_errors:
                avg_hw = np.mean(hw_sync_errors)
                max_hw = np.max(hw_sync_errors)
                std_hw = np.std(hw_sync_errors)
                print("  - Hardware Timestamp Synchronization Quality:")
                print(f"    - Average Sync Error: {avg_hw:.2f} ms")
                print(f"    - Max Sync Error: {max_hw:.2f} ms")
                print(f"    - Std Dev of Error: {std_hw:.2f} ms")
            print("---------------------------------\n")

        # Release video writers
        if 'video_writers' in locals():
            for vw in video_writers.values():
                vw.release()
            print("Saved videos.")
            
            # Extract frames immediately
            print("\n----- Extracting Frames from Videos -----")
            for pt_id in camera_mapping.values():
                video_path = f"{output_dir}/color/{pt_id}.mp4"
                images_dir = f"{output_dir}/color/{pt_id}"
                
                if not os.path.exists(video_path):
                    continue
                    
                print(f"Extracting Camera {pt_id}...")
                cap = cv2.VideoCapture(video_path)
                count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    cv2.imwrite(f"{images_dir}/{count}.png", frame)
                    count += 1
                cap.release()
                
                # Verify sync with depth
                depth_files = glob.glob(f"{output_dir}/depth/{pt_id}/*.npy")
                if len(depth_files) != count:
                    print(f"  WARNING: Sync mismatch! Video Frames: {count}, Depth Files: {len(depth_files)}")
                else:
                    print(f"  Camera {pt_id}: Extracted {count} frames (Sync Verified).")
            print("-----------------------------------------")

        # Save metadata
        with open(f"{output_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save calibration in PhysTwin format
        # Load existing calibrate.pkl and remap keys
        if os.path.exists("calibrate.pkl"):
            with open("calibrate.pkl", "rb") as f:
                old_calib = pickle.load(f)
            
            new_calib = {}
            for internal_key, matrix in old_calib.items():
                pt_id = camera_mapping[internal_key]
                new_calib[pt_id] = matrix
            
            with open(f"{output_dir}/calibrate.pkl", "wb") as f:
                pickle.dump(new_calib, f)
            
            print(f"Saved PhysTwin-compatible calibrate.pkl with keys: {list(new_calib.keys())}")
        
        print(f"Saved metadata to {output_dir}/metadata.json")
        print(f"\nTo use with PhysTwin:")
        print(f"  1. Copy {output_dir}/ to PhysTwin's data/different_types/my_object/")
        print(f"  2. Run: python script_process_data.py --case_name my_object")
            
        for dev in cam_sys.devices:
            dev.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="demo_data", help="Directory to save output data")
    args = parser.parse_args()
    main(args.output_dir)