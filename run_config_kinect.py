import cv2
import json
import time
import os
import numpy as np
import pyk4a
from pyk4a import PyK4A, Config, ColorResolution, DepthMode, FPS

# ==============================================================================
# ⚙️ CONFIGURATION
# ==============================================================================
CONFIG_FILE = "config_kinect.json"
DISPLAY_SCALE = 0.50

# Kinect Exposure Steps (us)
EXPOSURE_STEPS = [500, 1250, 2500, 5000, 10000, 12500, 20000, 30000, 40000, 50000, 60000, 100000]

# ==============================================================================
# 🛠️ HELPER FUNCTIONS
# ==============================================================================
def load_config():
    cfg = {
        "WH": [1280, 720], "fps": 30,
        "depth_mode": "NFOV_UNBINNED",
        "color_resolution": "RES_720P",
        "exposure": 12500, "gain": 128,
        "visual_scale": 5,
        "min_dist": 0.20, "max_dist": 1.50,
        "history_fill": 1, "history_decay": 5
    }
    
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded = json.load(f)
                for k, v in loaded.items():
                    if k in cfg: cfg[k] = v
        except Exception as e:
            print(f"[WARN] Failed to load config: {e}")
    return cfg

def save_config(settings):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
        print(f"[SUCCESS] Saved to {CONFIG_FILE}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save: {e}")
        return False

def map_exposure(val_idx):
    idx = int(np.clip(val_idx, 0, len(EXPOSURE_STEPS) - 1))
    return EXPOSURE_STEPS[idx]

def get_exposure_idx(val):
    # Find closest exposure step index
    return min(range(len(EXPOSURE_STEPS)), key=lambda i: abs(EXPOSURE_STEPS[i] - val))

def nothing(val): pass

# ==============================================================================
# 🚀 MAIN LOOP
# ==============================================================================
def run():
    print(f"[INFO] Starting Kinect Tuner...")
    p = load_config()

    # 1. Setup Kinect Config
    # Map string modes to PyK4A enums
    depth_mode_enum = getattr(DepthMode, p["depth_mode"], DepthMode.NFOV_UNBINNED)
    color_res_enum = getattr(ColorResolution, p["color_resolution"], ColorResolution.RES_720P)
    fps_enum = FPS.FPS_30 if p["fps"] == 30 else FPS.FPS_15

    k4a_config = Config(
        color_resolution=color_res_enum,
        depth_mode=depth_mode_enum,
        camera_fps=fps_enum,
        synchronized_images_only=True,
    )

    # 2. Start Camera
    k4a = PyK4A(config=k4a_config)
    k4a.start()

    # Initial Hardware Set
    k4a.exposure_mode_auto = False
    k4a.exposure = int(p["exposure"])
    k4a.gain = int(p["gain"])

    # 3. UI Setup
    window_name = "Kinect Tuner"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # Sliders
    # Exposure (Index based for Kinect)
    init_exp_idx = get_exposure_idx(p["exposure"])
    cv2.createTrackbar("Exp Index", window_name, init_exp_idx, len(EXPOSURE_STEPS) - 1, nothing)
    cv2.createTrackbar("Gain", window_name, int(p["gain"]), 255, nothing)
    
    # Range (CM)
    cv2.createTrackbar("Min Dist(cm)", window_name, int(p["min_dist"]*100), 200, nothing)
    cv2.createTrackbar("Max Dist(cm)", window_name, int(p["max_dist"]*100), 400, nothing)
    
    # Visualization
    cv2.createTrackbar("Vis Scale", window_name, int(p["visual_scale"]), 50, nothing)

    # History (Software Filter)
    cv2.createTrackbar("HIST FILL", window_name, int(p["history_fill"]), 1, nothing)
    cv2.createTrackbar("HIST DECAY", window_name, int(p["history_decay"]), 200, nothing)

    # State
    hist_buf = None
    age_buf = None
    bg_buf = None
    prev_hw = None # To track hardware changes
    msg_timer = 0

    try:
        while True:
            # --- Read Sliders ---
            exp_idx = cv2.getTrackbarPos("Exp Index", window_name)
            gain_val = max(0, cv2.getTrackbarPos("Gain", window_name))
            
            min_cm = cv2.getTrackbarPos("Min Dist(cm)", window_name)
            max_cm = max(min_cm+1, cv2.getTrackbarPos("Max Dist(cm)", window_name))
            
            vis_scale_int = max(1, cv2.getTrackbarPos("Vis Scale", window_name))
            
            use_hist = cv2.getTrackbarPos("HIST FILL", window_name)
            hist_decay = cv2.getTrackbarPos("HIST DECAY", window_name)

            # --- Update Hardware (Only if changed) ---
            exposure_us = map_exposure(exp_idx)
            cur_hw = (exposure_us, gain_val)
            
            if prev_hw is None or cur_hw != prev_hw:
                try:
                    k4a.exposure = int(exposure_us)
                    k4a.gain = int(gain_val)
                except Exception as e:
                    print(f"[WARN] Set HW failed: {e}")
                
                # Reset history on significant change
                hist_buf = None
                prev_hw = cur_hw

            # --- Capture ---
            capture = k4a.get_capture()
            if capture.color is None or capture.depth is None:
                continue

            # --- Processing ---
            # 1. Get Data
            color_img = capture.color[:, :, :3]  # Remove Alpha channel if present
            # transformed_depth aligns depth to color camera viewpoint
            raw_depth = capture.transformed_depth 
            
            if raw_depth is None: continue

            # 2. Smart History Fill (Software Filter Simulation)
            if use_hist:
                if hist_buf is None:
                    hist_buf = np.zeros_like(raw_depth)
                    age_buf = np.zeros_like(raw_depth, dtype=np.uint16)
                    bg_buf = np.zeros_like(raw_depth)
                
                valid = raw_depth > 0
                bg_buf[valid] = np.maximum(bg_buf[valid], raw_depth[valid])
                hist_buf[valid] = raw_depth[valid]
                age_buf[valid] = 0
                age_buf[~valid] += 1
                
                if hist_decay > 0:
                    decay = age_buf > hist_decay
                    hist_buf[decay] = bg_buf[decay]
                    age_buf[decay] = 0
                
                final_depth = hist_buf.copy()
            else:
                final_depth = raw_depth
                hist_buf = None

            # 3. Min/Max Range Clip
            # Kinect uses Millimeters (mm). 
            thresh_min_mm = min_cm * 10
            thresh_max_mm = max_cm * 10
            
            mask = (final_depth >= thresh_min_mm) & (final_depth <= thresh_max_mm)
            final_depth[~mask] = 0

            # --- Visualization ---
            # Vis scale for Kinect (mm). 
            # 5000mm * 0.05 = 250 (fits in 8-bit). 
            # Slider value 5 -> 0.05
            alpha = vis_scale_int / 100.0
            
            depth_cm = cv2.applyColorMap(
                cv2.convertScaleAbs(final_depth, alpha=alpha),
                cv2.COLORMAP_JET
            )
            depth_cm[final_depth == 0] = 0
            
            # Overlay
            overlay = cv2.addWeighted(color_img, 0.6, depth_cm, 0.4, 0)

            # Resize for Display
            w_disp = int(p["WH"][0] * DISPLAY_SCALE)
            h_disp = int(p["WH"][1] * DISPLAY_SCALE)
            display = cv2.resize(overlay, (w_disp, h_disp))

            # UI Text
            info_txt = f"Exp:{exposure_us}us Gain:{gain_val} Range:{min_cm}-{max_cm}cm"
            cv2.putText(display, info_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if msg_timer > 0:
                cv2.putText(display, "SAVED!", (w_disp//2 - 40, h_disp//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                msg_timer -= 1
            else:
                cv2.putText(display, "Press 'S' to Save Config", (10, h_disp - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(window_name, display)

            # --- Input ---
            key = cv2.waitKey(1)
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                out_config = {
                    "WH": p["WH"], "fps": p["fps"],
                    "depth_mode": p["depth_mode"],
                    "color_resolution": p["color_resolution"],
                    "exposure": int(exposure_us),
                    "gain": int(gain_val),
                    "visual_scale": int(vis_scale_int),
                    "min_dist": float(min_cm) / 100.0, # Save as meters
                    "max_dist": float(max_cm) / 100.0,
                    "history_fill": int(use_hist),
                    "history_decay": int(hist_decay)
                }
                if save_config(out_config):
                    msg_timer = 30

    finally:
        k4a.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
