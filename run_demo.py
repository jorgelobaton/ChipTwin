import pyrealsense2 as rs
from pyk4a import PyK4A, connected_device_count
import time
import os
import cv2
import numpy as np
import json
import pickle

# Import your CameraSystem
try:
    from qqtt.env.camera.camera_system import CameraSystem
except ImportError:
    try:
        from camera_system import CameraSystem
    except ImportError:
        print("Error: Could not import CameraSystem.")
        exit(1)

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

def main():
    rs_serials, d405_serials, k_indices = discover_cameras()
    
    if not (rs_serials or d405_serials or k_indices):
        print("\nNo cameras detected! Please check USB connections.")
        return

    # Configuration
    rs_config = {"WH": [848, 480], "fps": 30, "serials": rs_serials}
    d405_config = {"WH": [1280, 720], "fps": 30, "serials": d405_serials}
    kinect_config = {"mode": "WFOV_BINNED", "fps": 30, "indices": k_indices}

    print("\n--- Initializing System ---")
    cam_sys = CameraSystem(
        realsense_config=rs_config,
        d405_config=d405_config,
        kinect_config=kinect_config,
        exposure=10000, 
        gain=60
    )

    # --- PHYSTWIN-COMPATIBLE DATA STRUCTURE ---
    output_dir = "demo_data"
    
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
    
    for pt_id in camera_mapping.values():
        os.makedirs(f"{output_dir}/color/{pt_id}", exist_ok=True)
        os.makedirs(f"{output_dir}/depth/{pt_id}", exist_ok=True)
    
    # Metadata structure matching PhysTwin format
    metadata = {
        "recording": {str(pt_id): {} for pt_id in camera_mapping.values()},
        "intrinsics": [],
        "serial_numbers": [],
        "fps": 30,
        "WH": [1280, 720]  # Will be overridden per camera if needed
    }
    
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

    print(f"\nSystem Ready.")
    print(f"Press 'Space' to START/PAUSE recording to '{output_dir}/'")
    print(f"Press 'Esc' to exit and save metadata.")
    print(f"Recording will save as: color/0/, color/1/, etc.")

    try:
        while True:
            obs = cam_sys.get_observation()
            if not obs: continue
            
            # 1. PREVIEW LOOP
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

            # 2. RECORDING LOGIC
            if cam_sys.recording:
                if not is_recording_active:
                    print(f">> Recording STARTED (Frame {global_frame_count})")
                    is_recording_active = True
                
                # Check if we have new data from ALL cameras
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
                        
                        # Save with integer frame ID
                        cv2.imwrite(f"{output_dir}/color/{pt_id}/{global_frame_count}.png", 
                                   data["color"])
                        np.save(f"{output_dir}/depth/{pt_id}/{global_frame_count}.npy", 
                               data["depth"])
                        
                        # Log timestamp
                        metadata["recording"][str(pt_id)][global_frame_count] = ts
                    
                    global_frame_count += 1
                    
                    # Print progress every 30 frames
                    if global_frame_count % 30 == 0:
                        print(f"  Recorded {global_frame_count} frames...")
            
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
    main()
