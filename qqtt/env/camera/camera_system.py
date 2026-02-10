from .realsense import SingleRealsense 
from .single_kinect import SingleKinect
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import time
from pynput import keyboard
import cv2
import json
import os
import pickle

# Increase print options for debugging
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)

def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

class MultiCamera:
    """
    Backend manager that spawns specific worker processes for different camera types.
    """
    def __init__(self, rs_config, config_d405, config_kinect, shm_manager, name_mapping=None):
        self.cameras = {}
        self.devices_list = [] # For accessing device-specific methods like get_intrinsics()
        self.name_mapping = name_mapping or {} # Serial -> Human-readable ID
        self.d405_names = []
        self.kinect_names = []
        self.rs_names = []

        # 1. Standard RealSense (D415/D435/D455)
        if rs_config:
            serials = rs_config.get("serials", [])
            wh = rs_config.get("WH", (848, 480))
            fps = rs_config.get("fps", 30)
            advanced_mode_config = rs_config.get("advanced_mode_config", None)
            if advanced_mode_config is None and "parameters" in rs_config:
                advanced_mode_config = rs_config
            
            for serial in serials:
                print(f"[System] Init Standard RealSense: {serial}")
                cam = SingleRealsense(
                    shm_manager=shm_manager,
                    serial_number=serial,
                    resolution=tuple(wh),
                    capture_fps=fps,
                    enable_color=True,
                    enable_depth=True,
                    process_depth=False,
                    advanced_mode_config=advanced_mode_config,
                    verbose=False
                )
                name = self.name_mapping.get(serial, serial)
                self.cameras[name] = cam
                self.devices_list.append(cam)
                self.rs_names.append(name)

        # 2. RealSense D405
        if config_d405:
            serials = config_d405.get("serials", [])
            wh = config_d405.get("WH", (1280, 720))
            fps = config_d405.get("fps", 30)
            preset_id = config_d405.get("preset_id", 4)
            disparity_transform = config_d405.get("disparity_transform", True)
            history_fill = config_d405.get("history_fill", True)
            history_decay = config_d405.get("history_decay", 30)
            spatial_filter = config_d405.get("spatial_filter", {})
            temporal_filter = config_d405.get("temporal_filter", {})
            advanced_mode_config = config_d405.get("advanced_mode_config", None)
            if advanced_mode_config is None and "parameters" in config_d405:
                advanced_mode_config = config_d405
            
            for serial in serials:
                print(f"[System] Init D405 (High Density): {serial}")
                cam = SingleRealsense(
                    shm_manager=shm_manager,
                    serial_number=serial,
                    resolution=tuple(wh),
                    capture_fps=fps,
                    enable_color=True,
                    enable_depth=True,
                    process_depth=True, # ACTIVATES D405 TUNER FILTERS
                    preset_id=preset_id,
                    disparity_transform=disparity_transform,
                    history_fill=history_fill,
                    history_decay=history_decay,
                    spatial_filter=spatial_filter,
                    temporal_filter=temporal_filter,
                    advanced_mode_config=advanced_mode_config,
                    verbose=False
                )
                name = self.name_mapping.get(serial, serial)
                self.cameras[name] = cam
                self.devices_list.append(cam)
                self.d405_names.append(name)

        # 3. Azure Kinect
        if config_kinect:
            indices = config_kinect.get("indices", [])
            wh = config_kinect.get("WH", (1280, 720))
            fps = config_kinect.get("fps", 30)
            
            # Map indices to serials for consistent keying
            available_kinect_serials = SingleKinect.get_connected_devices_serial()
            
            for idx in indices:
                if idx < len(available_kinect_serials):
                    serial = available_kinect_serials[idx]
                    print(f"[System] Init Azure Kinect: {serial} (Index {idx})")
                    cam = SingleKinect(
                        shm_manager=shm_manager,
                        serial_number=serial,
                        resolution=tuple(wh),
                        capture_fps=fps,
                        enable_color=True,
                        enable_depth=True,
                        verbose=False
                    )
                    # Use provided name, else fallback to kinect_X
                    name = self.name_mapping.get(serial, f"kinect_{idx}")
                    self.cameras[name] = cam
                    self.kinect_names.append(name)
                    
                    # Store device index on object for the demo script's intrinsic lookup
                    cam.device_index = idx 
                    self.devices_list.append(cam)
                else:
                    print(f"[Error] Kinect Index {idx} out of range.")

    def start(self):
        for cam in self.cameras.values():
            cam.start()

    def stop(self):
        for cam in self.cameras.values():
            cam.stop()

    def is_ready(self):
        return all(cam.is_ready for cam in self.cameras.values())

    def get(self, k=None):
        results = {}
        for name, cam in self.cameras.items():
            try:
                data = cam.get(k)
                if data is not None:
                    results[name] = data
            except:
                continue
        return results if results else None

    def set_exposure(self, exposure=None, gain=None):
        """
        Set exposure and gain.
        exposure/gain can be:
        - a single value (applied to all)
        - a dict {serial_or_name: value}
        """
        for name, cam in self.cameras.items():
            if isinstance(cam, (SingleRealsense, SingleKinect)):
                # Determine exposure for this specific camera
                c_exp = exposure
                if isinstance(exposure, dict):
                    # Try name first, then serial
                    c_exp = exposure.get(name, exposure.get(cam.serial_number))
                
                # Determine gain for this specific camera
                c_gain = gain
                if isinstance(gain, dict):
                    c_gain = gain.get(name, gain.get(cam.serial_number))
                
                cam.set_exposure(c_exp, c_gain)

class CameraSystem:
    def __init__(
        self, 
        realsense_config=None,
        config_d405=None,
        config_kinect=None,
        exposure=None, 
        gain=None,
        name_mapping=None
    ):
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        
        # Initialize Backend
        self.cameras = MultiCamera(
            rs_config=realsense_config,
            config_d405=config_d405,
            config_kinect=config_kinect,
            shm_manager=self.shm_manager,
            name_mapping=name_mapping
        )

        # Store D405 names for rotation/intrinsic transformation
        self.d405_names = self.cameras.d405_names
        self.kinect_names = self.cameras.kinect_names

        self.cameras.start()

        # Apply settings after cameras are started
        self.cameras.set_exposure(exposure=exposure, gain=gain)
        
        # Public property for external scripts to access device list
        self.devices = self.cameras.devices_list
        
        self.recording = False
        self.end = False
        
        # Optional: Internal keyboard listener (can be ignored if run_demo handles it)
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    @property
    def num_cam(self):
        return len(self.cameras.cameras)

    def get_observation(self):
        obs = self._get_sync_frame()
        if obs is None:
            return None
        
        # Transformation: Rotate D405 and Crop all to 720x720
        transformed_obs = {}
        for name, data in obs.items():
            color = data['color']
            depth = data['depth']
            
            # 1. Rotate D405 90deg Clockwise
            if name in self.d405_names:
                color = cv2.rotate(color, cv2.ROTATE_90_CLOCKWISE)
                depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
            
            # 2. Center Crop to 720x720
            h, w = color.shape[:2]
            target_h, target_w = 720, 720
            
            y_start = max(0, (h - target_h) // 2)
            x_start = max(0, (w - target_w) // 2)
            
            # Crop if larger than target
            if h > target_h or w > target_w:
                color = color[y_start:y_start+target_h, x_start:x_start+target_w]
                depth = depth[y_start:y_start+target_h, x_start:x_start+target_w]
            
            transformed_obs[name] = {
                "color": color,
                "depth": depth,
                "timestamp": data['timestamp'],
                "step_idx": data['step_idx'],
                "camera_receive_timestamp": data.get("camera_receive_timestamp", 0)
            }
            
        return transformed_obs

    def get_intrinsics(self, name):
        """
        Returns the transformed intrinsic matrix considering the rotation and cropping.
        """
        cam = self.cameras.cameras.get(name)
        if not cam:
            return None
            
        K = cam.get_intrinsics().copy()
        
        # 1. Handle Rotation (D405)
        if name in self.d405_names:
            # Assuming D405 starts as 1280x720
            W_orig = 1280 
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            # After CW rotation: x'=y, y'=W-x
            # K' = [fy 0 cy; 0 fx W-cx; 0 0 1]
            K = np.array([
                [fy, 0, cy],
                [0, fx, W_orig - cx],
                [0, 0, 1]
            ], dtype=np.float64)
            h_rot, w_rot = 1280, 720
        else:
            # No rotation, get current resolution
            # Standard RS: 848x480, Kinect: 1280x720
            if name in self.kinect_names:
                h_rot, w_rot = 720, 1280
            else:
                # Fallback to whatever the camera reports if possible, 
                # but we usually know the config
                h_rot, w_rot = cam.resolution[1], cam.resolution[0]

        # 2. Handle Cropping to 720x720
        target_h, target_w = 720, 720
        y_start = max(0, (h_rot - target_h) // 2)
        x_start = max(0, (w_rot - target_w) // 2)
        
        K[0, 2] -= x_start
        K[1, 2] -= y_start
        
        return K

    def get_depth_scale(self, name):
        cam = self.cameras.cameras.get(name)
        if not cam:
            return 0.001
        if hasattr(cam, "get_depth_scale"):
            return cam.get_depth_scale()
        return 0.001

    def _get_sync_frame(self, k=4):
        # Get data from all cameras
        data_map = self.cameras.get(k=k)
        if not data_map or len(data_map) < len(self.cameras.cameras): 
            return None

        # Robust Synchronization Logic
        # 1. Collect all latest timestamps
        latest_timestamps = []
        valid_keys = []
        
        for key, val in data_map.items():
            if val is not None and len(val['timestamp']) > 0:
                latest_timestamps.append(val['timestamp'][-1])
                valid_keys.append(key)
        
        if not latest_timestamps: return None

        # 2. Determine anchor timestamp (median is robust against one laggy camera)
        target_ts = np.median(latest_timestamps)

        result = {}
        for key in valid_keys:
            val = data_map[key]
            timestamps = val["timestamp"]
            
            # Find index with minimum time distance to target
            dists = np.abs(timestamps - target_ts)
            best_idx = np.argmin(dists)
            
            # Threshold check: if sync is off by > 50ms, warn or skip?
            # For now, we take the best we have.
            
            result[key] = {
                "color": val["color"][best_idx],
                "depth": val["depth"][best_idx],
                "timestamp": val["timestamp"][best_idx],
                "step_idx": val["step_idx"][best_idx],
                # Pass through receive timestamp for analysis
                "camera_receive_timestamp": val.get("camera_receive_timestamp", 0) 
            }

        return result

    def on_press(self, key):
        try:
            if key == keyboard.Key.space:
                self.recording = not self.recording
                print(f"Recording: {self.recording}")
            elif key == keyboard.Key.esc:
                self.end = True
        except AttributeError:
            pass

    def stop(self):
        self.cameras.stop()
        self.listener.stop()
        self.shm_manager.shutdown()

    def calibrate(self, visualize=True):
        import pickle
        # ChArUco setup based on generate_charuco.py parameters
        squares_x = 4
        squares_y = 5
        square_length = 0.05
        marker_length = 0.037
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
        detector = cv2.aruco.CharucoDetector(board)

        print("\n--- Calibration Started ---")
        print("Waiting for valid ChArUco detection in all cameras...")
        print("CONTROL: [SPACE] Save Calibration, [ESC] Cancel/Exit")
        
        self.recording = False # Reset recording flag to use as Save trigger

        while not self.end:
            obs = self.get_observation()
            if obs is None: continue
            
            display_imgs = []
            camera_keys = sorted(obs.keys())
            
            poses = {} # key -> T_cam_board
            detections_complete = True
            
            for serial in camera_keys:
                data = obs[serial]
                img = data['color'].copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
                
                K = self.get_intrinsics(serial)
                
                if charuco_corners is not None and len(charuco_corners) >= 6:
                    obj_pts, img_pts = board.matchImagePoints(charuco_corners, charuco_ids)
                    success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, None)
                    
                    if success:
                        # Calculate Reprojection Error
                        proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
                        err = np.linalg.norm(img_pts - proj_pts.reshape(-1, 2), axis=1).mean()
                        
                        R, _ = cv2.Rodrigues(rvec)
                        T_cam_board = np.eye(4)
                        T_cam_board[:3, :3] = R
                        T_cam_board[:3, 3] = tvec.flatten()
                        poses[serial] = T_cam_board
                        
                        if visualize:
                            cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
                            cv2.drawFrameAxes(img, K, None, rvec, tvec, 0.1)
                            cv2.putText(img, f"RMS Error: {err:.3f} px", (30, 50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        detections_complete = False
                        if visualize:
                            cv2.putText(img, "PnP Failed", (30, 50), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    detections_complete = False
                    if visualize:
                        cv2.putText(img, "No Board Detected", (30, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if visualize:
                    display_imgs.append(img)
            
            if visualize:
                # Resize and combine for preview
                scaled_imgs = [cv2.resize(im, (640, 640)) for im in display_imgs]
                combined = np.hstack(scaled_imgs)
                
                status_text = "READY TO SAVE (Press SPACE)" if (detections_complete and len(poses) == self.num_cam) else "WAITING FOR ALL CAMERAS..."
                status_color = (0, 255, 0) if (detections_complete and len(poses) == self.num_cam) else (0, 0, 255)
                cv2.putText(combined, status_text, (50, combined.shape[0]-50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
                
                cv2.imshow("Calibration Preview", combined)
                # We check waitKey for space too in case listener is slow
                key = cv2.waitKey(1)
                if key == 27: self.end = True
                if key == 32: self.recording = True

            # If space was pressed and we have a valid set of poses
            if self.recording and len(poses) == self.num_cam:
                print("\nSaving Calibration...")
                # Anchor at first camera
                anchor_key = camera_keys[0]
                anchor_T_board = poses[anchor_key]
                final_c2ws = []
                c2ws_by_id = {}
                
                for serial in camera_keys:
                    # T_anchor_cam = T_anchor_board * T_board_cam
                    T_cam_board = poses[serial]
                    T_anchor_cam = anchor_T_board @ np.linalg.inv(T_cam_board)
                    final_c2ws.append(T_anchor_cam)
                    c2ws_by_id[serial] = T_anchor_cam
                
                # Save just the list of matrices to match existing pipeline
                with open("calibrate.pkl", "wb") as f:
                    pickle.dump(final_c2ws, f)
                    
                print(f"Success! Calibration saved to calibrate.pkl (Anchor: {anchor_key})")
                cv2.destroyAllWindows()
                return True
            
            # Reset recording flag if we failed to save (e.g. board lost right as space was pressed)
            if self.recording and len(poses) != self.num_cam:
                self.recording = False
        
        cv2.destroyAllWindows()
        return False
