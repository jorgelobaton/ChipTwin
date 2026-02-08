import cv2
import json
import numpy as np
import pyrealsense2 as rs  # <--- IMPORT RS HERE
from cams.camera_d405 import D405Camera

import argparse
import sys
import os

def save_config(cam):
    try:
        # Use cam.config_file instance variable
        with open(cam.config_file, 'w') as f:
            json.dump(cam.config_data, f, indent=4)
        print(f"[INFO] Saved to {cam.config_file}")
        return True
    except Exception as e:
        print(f"[ERR] Save failed: {e}")
        return False

def nothing(val): pass

def run_camera(serial, config_file):
    print(f"[INFO] Starting D405 Tuner for Serial: {serial} using {config_file}...")
    
    # Init Camera
    try:
        cam = D405Camera(serial_number=serial, config_file=config_file)
    except Exception as e:
        print(f"[ERR] Failed to open camera {serial}: {e}")
        return

    p = cam.config_data  # Reference to config dict
    
    cv2.namedWindow(f'D405 Tuner - {serial}', cv2.WINDOW_AUTOSIZE)

    # --- SLIDERS ---
    # Need to append serial to window name and trackbar names to be unique if running concurrently, 
    # but here we run sequentially, so it's fine. We use unique window name.
    win_name = f'D405 Tuner - {serial}'
    
    is_auto = 1 if p['exposure'] == -1 else 0
    exp_v = 5000 if p['exposure'] == -1 else p['exposure']
    
    cv2.createTrackbar('Auto Exp', win_name, is_auto, 1, nothing)
    cv2.createTrackbar('Exposure', win_name, int(exp_v), 33000, nothing)
    cv2.createTrackbar('Gain', win_name, int(p['gain']), 248, nothing)
    
    cv2.createTrackbar('Min Dist(cm)', win_name, int(p['min_dist']*100), 100, nothing)
    cv2.createTrackbar('Max Dist(cm)', win_name, int(p['max_dist']*100), 200, nothing)
    cv2.createTrackbar('Vis Scale', win_name, int(p['visual_scale']), 50, nothing)
    
    cv2.createTrackbar('USE DISPARITY', win_name, int(p['disparity_transform']), 1, nothing)
    
    s = p['spatial_filter']
    cv2.createTrackbar('S Enable', win_name, int(s['enable']), 1, nothing)
    cv2.createTrackbar('S Mag', win_name, int(s['magnitude']), 5, nothing)
    
    t = p['temporal_filter']
    cv2.createTrackbar('T Enable', win_name, int(t['enable']), 1, nothing)
    cv2.createTrackbar('T Persist', win_name, int(t['persistence']), 8, nothing)
    
    cv2.createTrackbar('HIST FILL', win_name, int(p['history_fill']), 1, nothing)
    cv2.createTrackbar('HIST DECAY', win_name, int(p['history_decay']), 200, nothing)

    prev_hw = None
    msg_timer = 0

    try:
        while True:
            # 1. Capture Frame
            depth, color = cam.get_frame()
            if depth is None: continue

            # 2. Read Sliders
            auto = cv2.getTrackbarPos('Auto Exp', win_name)
            exp = max(1, cv2.getTrackbarPos('Exposure', win_name))
            gain = max(16, cv2.getTrackbarPos('Gain', win_name))
            
            p['min_dist'] = cv2.getTrackbarPos('Min Dist(cm)', win_name) / 100.0
            p['max_dist'] = max(p['min_dist'] + 0.01, cv2.getTrackbarPos('Max Dist(cm)', win_name) / 100.0)
            vis = max(1, cv2.getTrackbarPos('Vis Scale', win_name))
            p['visual_scale'] = vis
            
            p['disparity_transform'] = cv2.getTrackbarPos('USE DISPARITY', win_name)
            
            s['enable'] = cv2.getTrackbarPos('S Enable', win_name)
            s['magnitude'] = max(1, cv2.getTrackbarPos('S Mag', win_name))
            
            t['enable'] = cv2.getTrackbarPos('T Enable', win_name)
            t['persistence'] = cv2.getTrackbarPos('T Persist', win_name)
            
            p['history_fill'] = cv2.getTrackbarPos('HIST FILL', win_name)
            p['history_decay'] = cv2.getTrackbarPos('HIST DECAY', win_name)

            # Update Thresholds
            cam.thresh_min = int(p['min_dist'] / cam.scale)
            cam.thresh_max = int(p['max_dist'] / cam.scale)

            # Hardware Update
            cur_hw = (auto, exp, gain)
            if prev_hw is None or cur_hw != prev_hw:
                p['exposure'] = -1 if auto else exp
                p['gain'] = gain
                cam._apply_hardware_settings()
                prev_hw = cur_hw
                
            # Filter Updates (Using 'rs' directly now)
            if s['enable']: cam.spatial.set_option(rs.option.filter_magnitude, s['magnitude'])
            if t['enable']: cam.temporal.set_option(rs.option.holes_fill, t['persistence'])

            # 3. Visualization
            d_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=vis/100.0), cv2.COLORMAP_JET)
            d_cm[depth == 0] = 0
            
            ov = cv2.addWeighted(color, 0.6, d_cm, 0.4, 0)
            
            disp_sz = int(720 * 0.6) 
            disp = cv2.resize(ov, (disp_sz, disp_sz))

            cv2.putText(disp, f"Cam: {serial}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(disp, f"{p['min_dist']:.2f}-{p['max_dist']:.2f}m", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            if msg_timer > 0:
                cv2.putText(disp, "SAVED!", (disp_sz//2-40, disp_sz//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                msg_timer-=1
            else:
                cv2.putText(disp, "S: Save & Next/Exit", (10, disp_sz-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow(win_name, disp)
            
            k = cv2.waitKey(1)
            if k in [ord('q'), 27]: # Quit all
                return False 
            elif k == ord('s'):
                if save_config(cam):
                    msg_timer = 30
                # Continue if saved, but user might want to proceed to next cam
                # Prompt implies: tune first, press s or q, run second.
                # Let's treat 'S' as Save & Continue, 'Q' as Abort/Quit
                return True # Proceed to next camera

    except KeyboardInterrupt: pass
    finally:
        try:
            cam.stop()
        except: pass
        try:
            cv2.destroyWindow(win_name)
        except: pass
    return True

def run():
    parser = argparse.ArgumentParser()
    # Optional overrides, but main mode is iterating map
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--serial", type=str, default=None, help="Camera Serial Number")
    args = parser.parse_args()

    # Mode 1: Specific single camera (via args)
    if args.serial:
        config_file = args.config
        if not config_file:
             # Try map lookup
             if os.path.exists("camera_map.json"):
                try:
                    with open("camera_map.json", 'r') as f:
                        cmap = json.load(f)
                        config_file = cmap.get(args.serial, "config_d405.json")
                except: config_file = "config_d405.json"
             else:
                 config_file = "config_d405.json"
        
        run_camera(args.serial, config_file)
        return

    # Mode 2: Iterate through camera_map.json
    if not os.path.exists("camera_map.json"):
        print("[ERR] camera_map.json not found, and no serial provided.")
        print("Please provide --serial OR create camera_map.json")
        return

    print("[INFO] Loading camera_map.json...")
    try:
        with open("camera_map.json", 'r') as f:
            cmap = json.load(f)
    except Exception as e:
        print(f"[ERR] Failed to load map: {e}")
        return

    # Sort serials just to have deterministic order
    serials = sorted(cmap.keys())
    
    for serial in serials:
        config_file = cmap[serial]
        print(f"\n>>> NEXT CAMERA: {serial} (Config: {config_file})")
        should_continue = run_camera(serial, config_file)
        if not should_continue:
            print("[INFO] Tuning aborted by user.")
            break
            
    print("[INFO] Tuning Sequence Completed.")

if __name__ == "__main__":
    run()
