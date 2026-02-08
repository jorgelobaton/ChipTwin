import cv2
import numpy as np
import time
import pickle
import sys
import os
import json

# Import our unified camera wrappers
from cams.camera_d405 import D405Camera
from cams.camera_kinect import KinectCamera
import pyrealsense2 as rs
from pyk4a import connected_device_count

# ==============================================================================
# DISCOVERY HELPER
# ==============================================================================
def discover_cameras():
    """Discover connected D405s and Kinects."""
    print("[INFO] Discovering Cameras...")
    d405_serials = []
    try:
        ctx = rs.context()
        for dev in ctx.query_devices():
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            if "D405" in name: d405_serials.append(serial)
    except: pass

    k_indices = []
    try:
        k_count = connected_device_count()
        k_indices = list(range(k_count))
    except: pass

    return d405_serials, k_indices

# ==============================================================================
# CALIBRATION SYSTEM CLASS
# ==============================================================================
class CalibrationSystem:
    def __init__(self):
        self.cameras = {}      # dict: name -> camera_object
        self.name_mapping = {} # dict: name -> friendly_name
        self.end = False
        self.recording = False

        # Load Camera Map
        camera_map = {}
        if os.path.exists("cams/camera_map.json"):
            try:
                with open("cams/camera_map.json", 'r') as f:
                    camera_map = json.load(f)
                print(f"[INFO] Loaded camera map with {len(camera_map)} entries.")
            except: pass

        # 1. Initialize Cameras
        d405s, kinects = discover_cameras()
        
        # Init D405s
        # Sort to ensure deterministic assignment if using index-based fallback (like run_demo)
        d405s.sort()
        
        for i, s in enumerate(d405s):
            try:
                # Determine Config
                if s in camera_map:
                    conf = camera_map[s]
                    # If not found at provided path, check under cams/
                    if not os.path.exists(conf) and os.path.exists(os.path.join("cams", conf)):
                        conf = os.path.join("cams", conf)
                elif i == 0: conf = "cams/config_d405_0.json" # Fallback matching run_demo logic
                elif i == 1: conf = "cams/config_d405_1.json"
                else: conf = "cams/config_d405.json"

                print(f"[INIT] D405_{s} using {conf}")
                cam = D405Camera(serial_number=s, config_file=conf)
                name = f"D405_{s}"
                self.cameras[name] = cam
                self.name_mapping[name] = f"D405 ({s[-4:]})"
                # print(f"[INIT] {name}") # Already verified in prev line
            except Exception as e:
                print(f"[ERR] {s}: {e}")

        # Init Kinects
        for idx in kinects:
            try:
                cam = KinectCamera(device_index=idx)
                name = f"Kinect_{idx}"
                self.cameras[name] = cam
                self.name_mapping[name] = f"Kinect {idx}"
                print(f"[INIT] {name}")
            except Exception as e:
                print(f"[ERR] Kinect {idx}: {e}")
        
        self.num_cam = len(self.cameras)
        if self.num_cam == 0:
            print("[FATAL] No cameras found.")
            sys.exit(1)

        # 2. Setup ChArUco
        # Matching your reference parameters exactly
        squares_x = 4
        squares_y = 5
        square_length = 0.05
        marker_length = 0.037
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, self.dictionary)
        self.detector = cv2.aruco.CharucoDetector(self.board)

    def get_observation(self):
        """
        Retrieves current frames from all cameras.
        Returns dict: { name: {'color': img, 'depth': img} } or None if any fail.
        """
        obs = {}
        for name, cam in self.cameras.items():
            d, c = cam.get_frame()
            if d is None or c is None: return None
            obs[name] = {'color': c, 'depth': d}
        return obs

    def get_intrinsics_matrix(self, name):
        """Fetches the 3x3 matrix from the camera object wrapper."""
        return self.cameras[name].get_intrinsics_matrix()

    def get_distortion_coeffs(self, name):
        """Fetches distortion coefficients from the camera object wrapper."""
        cam = self.cameras[name]
        if hasattr(cam, "get_distortion_coeffs"):
            return cam.get_distortion_coeffs()
        return np.zeros(5, dtype=np.float64)

    def run(self, visualize=True):
        print("\n--- Calibration Started ---")
        print(f"Cameras: {list(self.cameras.keys())}")
        print("Waiting for valid ChArUco detection in all cameras...")
        print("CONTROL: [SPACE] Save Calibration, [ESC] Cancel/Exit")
        
        self.recording = False 

        while not self.end:
            # 1. Capture
            obs = self.get_observation()
            if obs is None: continue
            
            display_imgs = []
            camera_keys = sorted(obs.keys())
            
            poses = {} # key -> T_cam_board
            detections_complete = True
            
            # 2. Process Each Camera
            for name in camera_keys:
                data = obs[name]
                img = data['color'].copy()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                # Detect
                charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(gray)
                K = self.get_intrinsics_matrix(name)
                dist = self.get_distortion_coeffs(name)
                
                valid_pose = False

                if charuco_corners is not None and len(charuco_corners) >= 6:
                    # Sub-pixel refinement for better pose accuracy
                    try:
                        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                        cv2.cornerSubPix(
                            gray,
                            charuco_corners,
                            (5, 5),
                            (-1, -1),
                            criteria,
                        )
                    except Exception:
                        pass
                    obj_pts, img_pts = self.board.matchImagePoints(charuco_corners, charuco_ids)
                    
                    if obj_pts is not None and len(obj_pts) >= 4:
                        success, rvec, tvec = cv2.solvePnP(
                            obj_pts,
                            img_pts,
                            K,
                            dist,
                            flags=cv2.SOLVEPNP_ITERATIVE,
                        )
                        
                        if success:
                            # Optional refinement step for better accuracy
                            try:
                                rvec, tvec = cv2.solvePnPRefineLM(
                                    obj_pts, img_pts, K, dist, rvec, tvec
                                )
                            except Exception:
                                pass
                            valid_pose = True
                            
                            # Reprojection Error Calculation
                            proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
                            err = np.linalg.norm(img_pts - proj_pts.reshape(-1, 2), axis=1).mean()
                            
                            # Construct Pose Matrix
                            R, _ = cv2.Rodrigues(rvec)
                            T_cam_board = np.eye(4)
                            T_cam_board[:3, :3] = R
                            T_cam_board[:3, 3] = tvec.flatten()
                            poses[name] = T_cam_board
                            
                            if visualize:
                                cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
                                cv2.drawFrameAxes(img, K, dist, rvec, tvec, 0.1)
                                cv2.putText(img, f"RMSE: {err:.3f} px", (30, 50), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if not valid_pose:
                    detections_complete = False
                    if visualize:
                        status_msg = "PnP Failed" if (charuco_corners is not None and len(charuco_corners)>=6) else "No Board Detected"
                        cv2.putText(img, status_msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Label
                cv2.putText(img, self.name_mapping.get(name, name), (30, img.shape[0]-30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

                if visualize:
                    display_imgs.append(img)
            
            # 3. Visualization
            if visualize:
                # Resize for grid (640x640 per cam)
                scaled_imgs = [cv2.resize(im, (640, 640)) for im in display_imgs]
                combined = np.hstack(scaled_imgs)
                
                # Status Overlay
                ready = (detections_complete and len(poses) == self.num_cam)
                status_text = "READY TO SAVE (Press SPACE)" if ready else "WAITING FOR ALL CAMERAS..."
                status_color = (0, 255, 0) if ready else (0, 0, 255)
                
                # Add status bar at bottom
                bar_h = 60
                bar = np.zeros((bar_h, combined.shape[1], 3), dtype=np.uint8)
                cv2.putText(bar, status_text, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
                final_vis = np.vstack((combined, bar))
                
                cv2.imshow("Calibration Preview", final_vis)
                
                key = cv2.waitKey(1)
                if key == 27 or key == ord('q'): self.end = True
                if key == 32: self.recording = True

            # 4. Save Logic
            if self.recording and len(poses) == self.num_cam:
                print("\n[SAVE] Calculating Transforms...")
                
                # Anchor at first camera (sorted alphabetically by ID)
                anchor_key = camera_keys[0]
                anchor_T_board = poses[anchor_key]
                
                final_c2ws = []
                c2ws_by_id = {}
                
                print(f"   Anchor Camera: {anchor_key}")
                
                for serial in camera_keys:
                    # Calculate: T_anchor_cam = T_anchor_board * inv(T_cam_board)
                    T_cam_board = poses[serial]
                    T_anchor_cam = anchor_T_board @ np.linalg.inv(T_cam_board)
                    
                    final_c2ws.append(T_anchor_cam)
                    c2ws_by_id[serial] = T_anchor_cam
                
                # IMPORTANT: Transform to World Frame consistent with simulator
                # Simulator (reverse_z=True) expects objects to be in Negative Z.
                # Current Anchor calibration usually puts objects in Positive Z (if camera looks forward).
                # To fix this, we transform the World Frame such that Z is flipped.
                # Transform T_world_anchor = FlipZ (180 deg around X)
                # T_world_cam = T_world_anchor * T_anchor_cam
                
                T_flip = np.array([
                    [1, 0,  0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0,  0, 1]
                ]) # Rotate 180 around X
                
                final_c2ws_corrected = []
                for c2w in final_c2ws:
                    # c2w here is T_anchor_cam
                    # We want T_world_cam
                    final_c2ws_corrected.append(T_flip @ c2w)
                
                # Save as a list of c2w matrices (standard format for pipeline)
                # Note: The order corresponds to sorted(camera_ids), which matches metadata.json
                with open("calibrate.pkl", "wb") as f:
                    pickle.dump(final_c2ws_corrected, f)
                
                print(f"[SUCCESS] Calibration saved to calibrate.pkl (with World Z-Flip)")
                cv2.destroyAllWindows()
                return True
            
            # Reset recording if lost tracking
            if self.recording and len(poses) != self.num_cam:
                self.recording = False
        
        cv2.destroyAllWindows()
        return False

    def close(self):
        for cam in self.cameras.values():
            cam.stop()

if __name__ == "__main__":
    system = CalibrationSystem()
    try:
        system.run()
    finally:
        system.close()
