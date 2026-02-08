import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os
import open3d as o3d

class D405Camera:
    default_config_file = "config_d405.json"

    def __init__(self, serial_number=None, config_file=None):
        self.config_file = config_file if config_file else self.default_config_file
        self.config_data = self._load_config()
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        
        # Select specific device if serial provided
        if serial_number:
            self.rs_config.enable_device(serial_number)
            self.serial = serial_number
        else:
            # If no serial, we will fetch it after start
            self.serial = None
        
        # Stream Setup
        w, h = self.config_data['WH']
        fps = self.config_data['fps']
        self.rs_config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        self.rs_config.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        
        # Start Pipeline
        self.profile = self.pipeline.start(self.rs_config)
        self.device = self.profile.get_device()
        self.depth_sensor = self.device.first_depth_sensor()
        
        if self.serial is None:
            self.serial = self.device.get_info(rs.camera_info.serial_number)
        
        # Apply Hardware Settings
        self._apply_hardware_settings()
        
        # Get Scale & Init Filters (Standard Init Logic)
        self.scale = self.depth_sensor.get_depth_scale()
        self.thresh_min = int(self.config_data['min_dist'] / self.scale)
        self.thresh_max = int(self.config_data['max_dist'] / self.scale)
        self.align = rs.align(rs.stream.color)
        self.d2d = rs.disparity_transform(True)
        self.d2d_rev = rs.disparity_transform(False)
        self.spatial = rs.spatial_filter()
        self.temporal = rs.temporal_filter()
        self._init_filters()
        
        # Buffers
        self.hist_buf, self.age_buf, self.bg_buf = None, None, None

    def _load_config(self):
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f: return json.load(f)
        raise FileNotFoundError(f"Config {self.config_file} missing.")

    def _apply_hardware_settings(self):
        try:
            p = self.config_data
            self.depth_sensor.set_option(rs.option.visual_preset, p.get('preset_id', 4))
            
            # Sync Mode
            if 'inter_cam_sync_mode' in p:
                self.depth_sensor.set_option(rs.option.inter_cam_sync_mode, p['inter_cam_sync_mode'])
            
            if p.get('exposure', -1) == -1:
                self.depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
            else:
                self.depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
                self.depth_sensor.set_option(rs.option.exposure, p['exposure'])
                self.depth_sensor.set_option(rs.option.gain, p['gain'])
        except: pass

    def _init_filters(self):
        s = self.config_data['spatial_filter']
        if s['enable']:
            self.spatial.set_option(rs.option.filter_magnitude, s['magnitude'])
            self.spatial.set_option(rs.option.filter_smooth_alpha, s['alpha'])
            self.spatial.set_option(rs.option.filter_smooth_delta, s['delta'])
        t = self.config_data['temporal_filter']
        if t['enable']:
            self.temporal.set_option(rs.option.filter_smooth_alpha, t.get('alpha', 0.15))
            self.temporal.set_option(rs.option.filter_smooth_delta, t.get('delta', 10))
            self.temporal.set_option(rs.option.holes_fill, t['persistence'])

    def get_intrinsics_matrix(self):
        """Returns 3x3 Intrinsics Matrix (K) for OpenCV, adjusted for rotation and crop."""
        # Use COLOR intrinsics because calibration runs on color frames
        raw = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        
        K = np.array([
            [raw.fx, 0, raw.ppx],
            [0, raw.fy, raw.ppy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        w_orig, h_orig = raw.width, raw.height  # 1280x720
        
        # 1. Apply Rotation
        rot_angle = self.config_data.get('rotation_angle', 0)
        # Fallback for backward compatibility
        if 'rotation_angle' not in self.config_data and self.config_data.get('rotation_90'):
            rot_angle = 90
            
        if rot_angle == 90:
            # 90 deg CW
            # New K: cx' = H - cy, cy' = cx
            fx, fy, cx, cy = raw.fx, raw.fy, raw.ppx, raw.ppy
            K = np.array([
                [fy, 0, h_orig - cy],
                [0, fx, cx],
                [0, 0, 1]
            ], dtype=np.float64)
            w_rot, h_rot = h_orig, w_orig
        elif rot_angle == 180:
            # 180 deg CW
            fx, fy, cx, cy = raw.fx, raw.fy, raw.ppx, raw.ppy
            K = np.array([
                [fx, 0, w_orig - cx],
                [0, fy, h_orig - cy],
                [0, 0, 1]
            ], dtype=np.float64)
            w_rot, h_rot = w_orig, h_orig
        elif rot_angle == 270:
            # 270 deg CW (90 CCW)
            # New K: cx' = cy, cy' = W - cx
            fx, fy, cx, cy = raw.fx, raw.fy, raw.ppx, raw.ppy
            K = np.array([
                [fy, 0, cy],
                [0, fx, w_orig - cx],
                [0, 0, 1]
            ], dtype=np.float64)
            w_rot, h_rot = h_orig, w_orig
        else:
            w_rot, h_rot = w_orig, h_orig
            
        # 2. Apply Crop (Center 720x720)
        if self.config_data.get('crop_square'):
            target = 720
            y_start = max(0, (h_rot - target) // 2)
            x_start = max(0, (w_rot - target) // 2)
            
            K[0, 2] -= x_start  # cx
            K[1, 2] -= y_start  # cy
            
        return K

    def get_distortion_coeffs(self):
        """Returns distortion coefficients for the color stream (k1, k2, p1, p2, k3)."""
        raw = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        coeffs = np.array(raw.coeffs, dtype=np.float64)
        if coeffs.size >= 5:
            return coeffs[:5]
        if coeffs.size > 0:
            padded = np.zeros(5, dtype=np.float64)
            padded[:coeffs.size] = coeffs
            return padded
        return np.zeros(5, dtype=np.float64)

    def get_intrinsics(self):
        """Returns Open3D Pinhole Intrinsic Object (Compatible with run_demo.py)"""
        K = self.get_intrinsics_matrix()
        w, h = 720, 720 # Since we force crop square
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        return o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    def get_frame(self):
        fs = self.pipeline.wait_for_frames()
        fs = self.align.process(fs)
        df, cf = fs.get_depth_frame(), fs.get_color_frame()
        if not df or not cf: return None, None

        f = df
        if self.config_data.get('disparity_transform'): f = self.d2d.process(f)
        if self.config_data['spatial_filter']['enable']: f = self.spatial.process(f)
        if self.config_data['temporal_filter']['enable']: f = self.temporal.process(f)
        if self.config_data.get('disparity_transform'): f = self.d2d_rev.process(f)

        d_np = np.asanyarray(f.get_data())
        c_np = np.asanyarray(cf.get_data())

        rot_angle = self.config_data.get('rotation_angle', 0)
        # Fallback for backward compatibility
        if 'rotation_angle' not in self.config_data and self.config_data.get('rotation_90'):
            rot_angle = 90

        if rot_angle == 90:
            d_np = cv2.rotate(d_np, cv2.ROTATE_90_CLOCKWISE)
            c_np = cv2.rotate(c_np, cv2.ROTATE_90_CLOCKWISE)
        elif rot_angle == 180:
            d_np = cv2.rotate(d_np, cv2.ROTATE_180)
            c_np = cv2.rotate(c_np, cv2.ROTATE_180)
        elif rot_angle == 270:
            d_np = cv2.rotate(d_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
            c_np = cv2.rotate(c_np, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if self.config_data.get('crop_square'):
            h, w = d_np.shape
            sz = min(h, w)
            y0, x0 = (h-sz)//2, (w-sz)//2
            d_np = d_np[y0:y0+sz, x0:x0+sz]
            c_np = c_np[y0:y0+sz, x0:x0+sz]

        if self.config_data.get('history_fill'):
            if self.hist_buf is None or self.hist_buf.shape != d_np.shape:
                self.hist_buf = np.zeros_like(d_np)
                self.age_buf = np.zeros_like(d_np, dtype=np.uint16)
                self.bg_buf = np.zeros_like(d_np)
            v = d_np > 0
            self.bg_buf[v] = np.maximum(self.bg_buf[v], d_np[v])
            self.hist_buf[v] = d_np[v]
            self.age_buf[v] = 0
            self.age_buf[~v] += 1
            if self.config_data.get('history_decay', 0) > 0:
                dec = self.age_buf > self.config_data['history_decay']
                self.hist_buf[dec] = self.bg_buf[dec]
                self.age_buf[dec] = 0
            d_np = self.hist_buf.copy()

        mask = (d_np >= self.thresh_min) & (d_np <= self.thresh_max)
        d_np[~mask] = 0
        return d_np, c_np

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
