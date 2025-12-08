from .realsense import MultiRealsense, SingleRealsense
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import time
from pynput import keyboard
import cv2
import json
import os
import pickle


# --- PyK4A Imports ---
try:
    import pyk4a
    from pyk4a import Config, PyK4A, ColorResolution, DepthMode, FPS, CalibrationType, K4AException, K4ATimeoutException
    _kinect_available = True
except ImportError:
    print("Warning: 'pyk4a' not found. Azure Kinect will not work.")
    _kinect_available = False


np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


class AzureKinectWrapper:
    """
    Wraps Azure Kinect to match the interface used by CameraSystem.
    """
    def __init__(self, device_index, mode_str="WFOV_UNBINNED", fps=30):
        if not _kinect_available:
            raise ImportError("pyk4a library is missing. Install with 'pip install pyk4a'")

        self.device_index = device_index
        self.fps = fps
        
        self.MODE_MAP = {
            "NFOV_UNBINNED": DepthMode.NFOV_UNBINNED,
            "NFOV_BINNED": DepthMode.NFOV_2X2BINNED,
            "WFOV_BINNED": DepthMode.WFOV_2X2BINNED,
            "WFOV_UNBINNED": DepthMode.WFOV_UNBINNED,
            "PASSIVE_IR": DepthMode.PASSIVE_IR,
        }
        
        fps_map = {5: FPS.FPS_5, 15: FPS.FPS_15, 30: FPS.FPS_30}
        k4a_fps = fps_map.get(fps, FPS.FPS_30)

        self.k4a = PyK4A(
            Config(
                color_resolution=ColorResolution.RES_1536P,
                depth_mode=self.MODE_MAP.get(mode_str, DepthMode.WFOV_UNBINNED),
                camera_fps=k4a_fps,
                synchronized_images_only=True,
            ),
            device_id=device_index,
        )
        self.k4a.start()
        
        self.capture_history = []
        self.max_history = 30
        print(f"Azure Kinect (Index {device_index}) initialized in {mode_str}")

    def get(self, k=1):
        try:
            capture = self.k4a.get_capture()
        except K4ATimeoutException:
            capture = None
        except K4AException as e:
            print(f"Kinect Warning: {e}")
            capture = None

        if capture is not None and capture.color is not None and capture.depth is not None:
            # OpenCV usually expects BGR. PyK4A returns BGRA.
            color = capture.color[:, :, :3] 
            depth = capture.depth
            
            timestamp = capture.depth_timestamp_usec / 1e6
            
            self.capture_history.append({
                "color": color,
                "depth": depth,
                "timestamp": timestamp,
                "step_idx": 0 
            })
            if len(self.capture_history) > self.max_history:
                self.capture_history.pop(0)
        
        valid_k = min(k, len(self.capture_history))
        if valid_k == 0: return {}
        
        recent_frames = self.capture_history[-valid_k:]
        
        return {
            f"kinect_{self.device_index}": {
                "color": np.stack([f["color"] for f in recent_frames]),
                "depth": np.stack([f["depth"] for f in recent_frames]),
                "timestamp": np.array([f["timestamp"] for f in recent_frames]),
                "step_idx": np.array([0] * valid_k)
            }
        }
    
    def get_intrinsics(self):
        """
        Returns a list containing a tuple: (Camera Matrix, Distortion Coefficients).
        """
        try:
            mat = self.k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
            dist = self.k4a.calibration.get_distortion_coefficients(CalibrationType.COLOR)
            return [(mat, dist)] 
        except Exception as e:
            print(f"Error getting intrinsics for Kinect {self.device_index}: {e}")
            return [(np.eye(3), np.zeros(8))] 


    def stop(self):
        self.k4a.stop()


class CameraSystem:
    def __init__(
        self, 
        realsense_config={"WH": [848, 480], "fps": 30, "serials": []},
        d405_config={"WH": [848, 480], "fps": 30, "serials": []},
        kinect_config={"mode": "NFOV_UNBINNED", "fps": 30, "indices": []},
        exposure=10000, 
        gain=60, 
        white_balance=3800
    ):
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        
        self.devices = []
        self.cam_ids = [] 

        # 1. Initialize Standard RealSense Group
        if realsense_config.get("serials"):
            print(f"Initializing Standard RealSense: {realsense_config['serials']}")
            rs_group = MultiRealsense(
                serial_numbers=realsense_config["serials"],
                shm_manager=self.shm_manager,
                resolution=tuple(realsense_config["WH"]),
                capture_fps=realsense_config["fps"],
                enable_color=True, enable_depth=True, process_depth=True, verbose=False,
            )
            try:
                rs_group.set_exposure(exposure=exposure, gain=gain)
                rs_group.set_white_balance(white_balance)
            except Exception as e:
                print(f"Warning setting RS options: {e}")
                
            rs_group.start()
            self.devices.append(rs_group)
            self.cam_ids.extend(range(len(realsense_config["serials"])))

        # 2. Initialize D405 RealSense Group
        if d405_config.get("serials"):
            print(f"Initializing D405: {d405_config['serials']}")
            d405_group = MultiRealsense(
                serial_numbers=d405_config["serials"],
                shm_manager=self.shm_manager,
                resolution=tuple(d405_config["WH"]),
                capture_fps=d405_config["fps"],
                enable_color=True, enable_depth=True, process_depth=True, verbose=False,
            )
            try:
                d405_group.set_exposure(exposure=exposure, gain=gain)
                d405_group.set_white_balance(white_balance)
            except Exception as e:
                print(f"Warning setting D405 options: {e}")

            d405_group.start()
            self.devices.append(d405_group)

        # 3. Initialize Azure Kinects
        if kinect_config.get("indices"):
            for idx in kinect_config["indices"]:
                kinect = AzureKinectWrapper(
                    device_index=idx, 
                    mode_str=kinect_config.get("mode", "NFOV_UNBINNED"), 
                    fps=kinect_config.get("fps", 30)
                )
                self.devices.append(kinect)
                self.cam_ids.append(f"kinect_{idx}")

        time.sleep(3)
        self.recording = False
        self.end = False
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()
        print("Camera system is ready.")

    def get_observation(self):
        return self._get_sync_frame()

    def _get_sync_frame(self, k=4):
        raw_collections = []
        for dev in self.devices:
            raw_collections.append(dev.get(k=k))

        combined_data = {}
        rs_offset = 0
        
        for collection in raw_collections:
            for key, val in collection.items():
                new_key = key
                if isinstance(key, int):
                    new_key = key + rs_offset
                combined_data[new_key] = val
            
            if any(isinstance(k, int) for k in collection.keys()):
                rs_offset += len(collection)

        timestamps = []
        for cam_key in combined_data:
            if len(combined_data[cam_key]["timestamp"]) > 0:
                timestamps.append(combined_data[cam_key]["timestamp"][-1])
        
        if not timestamps: return None
        
        last_timestamp = np.min(timestamps)

        data = {}
        for cam_key, value in combined_data.items():
            this_timestamps = value["timestamp"]
            diffs = np.abs(this_timestamps - last_timestamp)
            best_idx = np.argmin(diffs)
            
            data[cam_key] = {}
            data[cam_key]["color"] = value["color"][best_idx]
            data[cam_key]["depth"] = value["depth"][best_idx]
            data[cam_key]["timestamp"] = value["timestamp"][best_idx]
            data[cam_key]["step_idx"] = value.get("step_idx", [0]*len(this_timestamps))[best_idx]

        return data

    def on_press(self, key):
        try:
            if key == keyboard.Key.space:
                if not self.recording:
                    self.recording = True
                    print("Space pressed (Action Triggered)")
                else:
                    self.recording = False
                    self.end = True
        except AttributeError:
            pass

    def record(self, output_path):
        exist_dir(output_path)
        exist_dir(f"{output_path}/color")
        exist_dir(f"{output_path}/depth")

        first_obs = self.get_observation()
        if not first_obs:
            print("No data received.")
            return

        self.cam_keys_cache = list(first_obs.keys())
        for key in self.cam_keys_cache:
            exist_dir(f"{output_path}/color/{key}")
            exist_dir(f"{output_path}/depth/{key}")

        metadata = {}
        metadata["recording"] = {str(k): {} for k in self.cam_keys_cache}
        
        all_intrinsics = []
        for dev in self.devices:
            try:
                ints = dev.get_intrinsics()
                processed_ints = []
                for x in ints:
                    if isinstance(x, tuple):
                        processed_ints.append(x[0]) 
                    else:
                        processed_ints.append(x)
                all_intrinsics.extend(processed_ints)
            except: pass
            
        metadata["intrinsics"] = [x.tolist() if hasattr(x, "tolist") else x for x in all_intrinsics]
        
        print(f"Recording cameras: {self.cam_keys_cache}")
        last_timestamps = {k: -1 for k in self.cam_keys_cache}
        
        while not self.end:
            if self.recording:
                obs = self.get_observation()
                if obs is None: continue

                for key in self.cam_keys_cache:
                    ts = obs[key]["timestamp"]
                    if ts != last_timestamps[key]:
                        last_timestamps[key] = ts
                        step_idx = obs[key]["step_idx"]
                        if step_idx == 0 or step_idx == -1:
                            step_idx = int(ts * 1000)

                        cv2.imwrite(f"{output_path}/color/{key}/{step_idx}.png", obs[key]["color"])
                        np.save(f"{output_path}/depth/{key}/{step_idx}.npy", obs[key]["depth"])
                        metadata["recording"][str(key)][step_idx] = ts

        print("End recording")
        self.listener.stop()
        with open(f"{output_path}/metadata.json", "w") as f:
            json.dump(metadata, f)
        
        for dev in self.devices:
            dev.stop()

    def calibrate(self, visualize=True):
        print("Initializing calibration board...")
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        
        board = cv2.aruco.CharucoBoard(
            (4, 5),
            squareLength=0.05,
            markerLength=0.037,
            dictionary=dictionary,
        )

        intrinsics_map = {}
        rs_offset = 0
        
        # Populate intrinsics map with (matrix, distortion) tuples
        for dev in self.devices:
            dev_intrinsics = dev.get_intrinsics() 
            
            if isinstance(dev, AzureKinectWrapper):
                key = f"kinect_{dev.device_index}"
                if len(dev_intrinsics) > 0:
                    intrinsics_map[key] = dev_intrinsics[0]
            else:
                for i, mat in enumerate(dev_intrinsics):
                    # Assume Realsense is rectified or low distortion (pass None)
                    intrinsics_map[rs_offset + i] = (mat, None)
                rs_offset += len(dev_intrinsics)
        
        print(f"Calibration Targets: {list(intrinsics_map.keys())}")
        print("Please hold the board visible to ALL cameras.")
        print("When 'READY: SPACE TO SAVE' appears, press SPACE to finish.")

        # Ensure recording flag is False to start
        self.recording = False

        while True:
            obs = self.get_observation()
            if not obs: continue
            
            missing_cams = [k for k in intrinsics_map.keys() if k not in obs]
            if missing_cams: continue
            
            c2ws = {} 
            fail_flag = False
            vis_images = {} # Accumulate images to show later
            
            # --- 1. Processing Loop ---
            for key in intrinsics_map.keys():
                image = obs[key]["color"]
                intrinsic, dist_coeffs = intrinsics_map[key]
                vis_img = image.copy()

                corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary)
                if ids is None or len(ids) == 0:
                    if visualize:
                        cv2.putText(vis_img, "NO MARKERS", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    fail_flag = True
                    vis_images[key] = vis_img
                    continue 
                
                retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, image, board, 
                    cameraMatrix=intrinsic,
                    distCoeffs=dist_coeffs
                )
                
                if charuco_corners is None or len(charuco_corners) < 11:
                    if visualize:
                        cv2.putText(vis_img, "FEW CORNERS", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                        if charuco_corners is not None:
                             cv2.aruco.drawDetectedCornersCharuco(vis_img, charuco_corners, charuco_ids)
                    fail_flag = True
                    vis_images[key] = vis_img
                    continue

                if visualize:
                    cv2.aruco.drawDetectedCornersCharuco(vis_img, charuco_corners, charuco_ids)

                retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    charuco_corners, 
                    charuco_ids, 
                    board, 
                    intrinsic, 
                    dist_coeffs, 
                    None, None 
                )
                
                if not retval:
                    fail_flag = True
                    vis_images[key] = vis_img
                    continue
                
                reprojected_points, _ = cv2.projectPoints(
                    board.getChessboardCorners()[charuco_ids, :],
                    rvec, tvec, intrinsic, dist_coeffs
                )
                error = np.sqrt(np.sum((reprojected_points.reshape(-1, 2) - charuco_corners.reshape(-1, 2)) ** 2, axis=1)).mean()
                
                if error > 1: 
                    if visualize:
                        cv2.putText(vis_img, f"HIGH ERROR: {error:.2f}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                    fail_flag = True
                    vis_images[key] = vis_img
                    continue
                
                if visualize:
                    cv2.putText(vis_img, f"OK ({error:.2f})", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.drawFrameAxes(vis_img, intrinsic, dist_coeffs, rvec, tvec, 0.1)

                R, _ = cv2.Rodrigues(rvec)
                w2c = np.eye(4)
                w2c[:3, :3] = R
                w2c[:3, 3] = tvec[:, 0]
                c2ws[key] = np.linalg.inv(w2c)
                vis_images[key] = vis_img

            # --- 2. Global Check & User Input ---
            all_cameras_good = (not fail_flag) and (len(c2ws) == len(intrinsics_map))

            # If user pressed space (self.recording=True) AND all cameras are good -> SAVE
            if all_cameras_good and self.recording:
                print("\nSUCCESS! Calibration complete for all cameras.")
                with open("calibrate.pkl", "wb") as f:
                    pickle.dump(c2ws, f)
                break
            
            # --- 3. Visualization Loop ---
            if visualize:
                for key in vis_images:
                    img = vis_images[key]
                    if all_cameras_good:
                        # Show prompt to save
                        cv2.putText(img, "READY: SPACE TO SAVE", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif self.recording:
                        # User pressed space but not ready
                        cv2.putText(img, "CANNOT SAVE YET", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    cv2.imshow(f"Calib {key}", img)
                    cv2.waitKey(1)
        
        cv2.destroyAllWindows()
        self.listener.stop()
        for dev in self.devices:
            dev.stop()
