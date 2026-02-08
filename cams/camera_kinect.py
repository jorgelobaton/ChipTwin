import cv2
import json
import os
import numpy as np
import pyk4a
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS
import open3d as o3d

class KinectCamera:
    CONFIG_FILE = "config_kinect.json"

    def __init__(self, device_index=0):
        self.config_data = self._load_config()
        self.device_index = device_index
        
        d_mode = getattr(DepthMode, self.config_data["depth_mode"], DepthMode.NFOV_UNBINNED)
        c_res = getattr(ColorResolution, self.config_data["color_resolution"], ColorResolution.RES_720P)
        fps = FPS.FPS_30 if self.config_data["fps"] == 30 else FPS.FPS_15
        
        self.k4a_config = Config(
            color_resolution=c_res, depth_mode=d_mode, camera_fps=fps,
            synchronized_images_only=True
        )
        
        self.k4a = PyK4A(self.k4a_config, device_id=device_index)
        self.k4a.start()
        
        self.k4a.exposure_mode_auto = False
        self.k4a.exposure = int(self.config_data["exposure"])
        self.k4a.gain = int(self.config_data["gain"])
        
        self.scale = self.config_data.get("depth_scale", 0.001)
        self.thresh_min = int(self.config_data["min_dist"] / self.scale)
        self.thresh_max = int(self.config_data["max_dist"] / self.scale)
        self.hist_buf, self.age_buf, self.bg_buf = None, None, None

    def _load_config(self):
        if os.path.exists(self.CONFIG_FILE):
            with open(self.CONFIG_FILE, 'r') as f: return json.load(f)
        return {
            "WH": [1280, 720], "fps": 30, "crop_square": 1,
            "depth_mode": "NFOV_UNBINNED", "color_resolution": "RES_720P",
            "exposure": 12500, "gain": 128, "visual_scale": 5, 
            "min_dist": 0.20, "max_dist": 1.50, "depth_scale": 0.001,
            "spatial_filter": 1, "spatial_sigma": 3, "hole_filling": 0,
            "history_fill": 0, "history_decay": 5,
            "edge_threshold": 50  # New: Gradient threshold for edge removal
        }

    def _remove_edge_artifacts(self, depth):
        """Invalidates 'flying pixels' at depth discontinuities like D405 behavior."""
        if depth is None: return None
        # Convert to float for gradient calculation
        depth_f = depth.astype(np.float32)
        
        # Calculate gradients in X and Y
        dx = cv2.Sobel(depth_f, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(depth_f, cv2.CV_32F, 0, 1, ksize=3)
        
        # Compute gradient magnitude
        mag = np.sqrt(dx**2 + dy**2)
        
        # Threshold: if change is too steep, it's a 'flying pixel' edge
        # Adjust 'edge_threshold' in config based on your scene (e.g., 50-100mm)
        edge_mask = mag > self.config_data.get("edge_threshold", 250)
        
        # Dilation expands the invalidation zone slightly to ensure clean separation
        kernel = np.ones((3,3), np.uint8)
        edge_mask = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1)
        
        depth[edge_mask > 0] = 0
        return depth

    def get_intrinsics_matrix(self):
        calib = self.k4a.calibration
        try:
            color_calib = calib._calibration.color_camera_calibration
            intr = color_calib.intrinsics.parameters.param
            fx, fy, cx, cy = intr.fx, intr.fy, intr.cx, intr.cy
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        except:
            K = calib.get_camera_matrix(pyk4a.calibration.CalibrationType.COLOR)
            
        w_orig, h_orig = 1280, 720
        if self.config_data.get("crop_square"):
            target = 720
            y_start = max(0, (h_orig - target) // 2)
            x_start = max(0, (w_orig - target) // 2)
            K[0, 2] -= x_start
            K[1, 2] -= y_start
        return K

    def get_distortion_coeffs(self):
        """Returns distortion coefficients (k1, k2, p1, p2, k3) for the color camera."""
        calib = self.k4a.calibration
        try:
            color_calib = calib._calibration.color_camera_calibration
            intr = color_calib.intrinsics.parameters.param
            k1 = getattr(intr, "k1", 0.0)
            k2 = getattr(intr, "k2", 0.0)
            p1 = getattr(intr, "p1", 0.0)
            p2 = getattr(intr, "p2", 0.0)
            k3 = getattr(intr, "k3", 0.0)
            return np.array([k1, k2, p1, p2, k3], dtype=np.float64)
        except Exception:
            return np.zeros(5, dtype=np.float64)

    def get_intrinsics(self):
        K = self.get_intrinsics_matrix()
        w, h = 720, 720 
        return o3d.camera.PinholeCameraIntrinsic(w, h, K[0,0], K[1,1], K[0,2], K[1,2])

    def get_frame(self):
        cap = self.k4a.get_capture()
        if cap.color is None or cap.depth is None: return None, None
        depth = cap.transformed_depth
        if depth is None: return None, None
        color = cap.color[:, :, :3]

        if self.config_data.get("crop_square"):
            h, w = depth.shape
            sz = min(h, w)
            x0 = (w - sz) // 2
            depth = depth[:, x0:x0+sz]
            color = color[:, x0:x0+sz]

        # 1. Remove edge artifacts FIRST (avoid smearing edges with filters)
        depth = self._remove_edge_artifacts(depth)

        # 2. Spatial Filter (Bilateral is better for preserving edges while smoothing)
        if self.config_data.get("spatial_filter"):
            sig = self.config_data.get("spatial_sigma", 3)
            d_fl = depth.astype(np.float32)
            # Bilateral filter helps remove ToF noise without blurring valid edges
            d_fl = cv2.bilateralFilter(d_fl, 5, sig*20, sig)
            depth = d_fl.astype(np.uint16)

        # 3. History Fill (Temporal filtering)
        if self.config_data.get("history_fill"):
            if self.hist_buf is None or self.hist_buf.shape != depth.shape:
                self.hist_buf, self.age_buf, self.bg_buf = np.zeros_like(depth), np.zeros_like(depth, dtype=np.uint16), np.zeros_like(depth)
            v = depth > 0
            self.bg_buf[v] = np.maximum(self.bg_buf[v], depth[v])
            self.hist_buf[v] = depth[v]
            self.age_buf[v] = 0
            self.age_buf[~v] += 1
            if self.config_data.get("history_decay", 0) > 0:
                dec = self.age_buf > self.config_data["history_decay"]
                self.hist_buf[dec] = self.bg_buf[dec]
                self.age_buf[dec] = 0
            depth = self.hist_buf.copy()

        # 4. Final Range Masking
        mask = (depth >= self.thresh_min) & (depth <= self.thresh_max)
        depth[~mask] = 0
        return depth, color

    def stop(self): self.k4a.stop()
