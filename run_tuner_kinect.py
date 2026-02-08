import cv2
import json
import numpy as np
from cams.camera_kinect import KinectCamera

# Helper to map slider index to exposure value
EXPOSURE_STEPS = [500, 1250, 2500, 5000, 10000, 12500, 20000, 30000, 40000, 50000, 60000, 100000]
def get_exp_idx(val): return min(range(len(EXPOSURE_STEPS)), key=lambda i: abs(EXPOSURE_STEPS[i]-val))
def map_exp(idx): return EXPOSURE_STEPS[int(np.clip(idx, 0, len(EXPOSURE_STEPS)-1))]

def save_config(cam):
    try:
        with open(cam.CONFIG_FILE, 'w') as f:
            json.dump(cam.config_data, f, indent=4)
        return True
    except: return False

def nothing(val): pass

def run():
    print("[INFO] Starting Kinect Tuner (Class-Based)...")
    cam = KinectCamera()
    p = cam.config_data
    
    cv2.namedWindow('Kinect Tuner', cv2.WINDOW_AUTOSIZE)

    init_exp = get_exp_idx(p["exposure"])
    cv2.createTrackbar("Exp Index", "Kinect Tuner", init_exp, len(EXPOSURE_STEPS)-1, nothing)
    cv2.createTrackbar("Gain", "Kinect Tuner", int(p["gain"]), 255, nothing)
    cv2.createTrackbar("Vis Scale", "Kinect Tuner", int(p["visual_scale"]), 50, nothing)
    
    cv2.createTrackbar("Min Dist(cm)", "Kinect Tuner", int(p["min_dist"]*100), 200, nothing)
    cv2.createTrackbar("Max Dist(cm)", "Kinect Tuner", int(p["max_dist"]*100), 400, nothing)

    cv2.createTrackbar("SPATIAL", "Kinect Tuner", int(p["spatial_filter"]), 1, nothing)
    cv2.createTrackbar("S Sigma", "Kinect Tuner", int(p.get("spatial_sigma", 3)), 10, nothing)
    cv2.createTrackbar("HOLE FILL", "Kinect Tuner", int(p["hole_filling"]), 1, nothing)
    cv2.createTrackbar("HIST FILL", "Kinect Tuner", int(p["history_fill"]), 1, nothing)
    cv2.createTrackbar("HIST DECAY", "Kinect Tuner", int(p["history_decay"]), 200, nothing)

    prev_hw = None
    msg_timer = 0

    try:
        while True:
            # 1. Capture
            depth, color = cam.get_frame()
            if depth is None: continue

            # 2. UI Updates
            exp_idx = cv2.getTrackbarPos("Exp Index", "Kinect Tuner")
            gain = max(0, cv2.getTrackbarPos("Gain", "Kinect Tuner"))
            vis = max(1, cv2.getTrackbarPos("Vis Scale", "Kinect Tuner"))
            
            p["visual_scale"] = vis
            p["spatial_filter"] = cv2.getTrackbarPos("SPATIAL", "Kinect Tuner")
            p["spatial_sigma"] = max(1, cv2.getTrackbarPos("S Sigma", "Kinect Tuner"))
            p["hole_filling"] = cv2.getTrackbarPos("HOLE FILL", "Kinect Tuner")
            p["history_fill"] = cv2.getTrackbarPos("HIST FILL", "Kinect Tuner")
            p["history_decay"] = cv2.getTrackbarPos("HIST DECAY", "Kinect Tuner")
            
            min_cm = cv2.getTrackbarPos("Min Dist(cm)", "Kinect Tuner")
            max_cm = max(min_cm+1, cv2.getTrackbarPos("Max Dist(cm)", "Kinect Tuner"))
            p["min_dist"] = min_cm / 100.0
            p["max_dist"] = max_cm / 100.0
            
            # Update Thresholds
            cam.thresh_min = int(p["min_dist"] / cam.scale)
            cam.thresh_max = int(p["max_dist"] / cam.scale)

            # Hardware Update
            exp_us = map_exp(exp_idx)
            cur_hw = (exp_us, gain)
            if prev_hw is None or cur_hw != prev_hw:
                try: 
                    cam.k4a.exposure = int(exp_us)
                    cam.k4a.gain = int(gain)
                    p["exposure"] = int(exp_us)
                    p["gain"] = int(gain)
                except: pass
                # Reset History on hw change
                cam.hist_buf = None
                prev_hw = cur_hw

            # 3. Viz
            d_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=vis/100.0), cv2.COLORMAP_JET)
            d_cm[depth==0] = 0
            ov = cv2.addWeighted(color, 0.6, d_cm, 0.4, 0)
            
            disp_sz = int(720 * 0.6)
            disp = cv2.resize(ov, (disp_sz, disp_sz))

            cv2.putText(disp, f"{p['min_dist']}-{p['max_dist']}m", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            if msg_timer > 0:
                cv2.putText(disp, "SAVED!", (disp_sz//2-40, disp_sz//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                msg_timer-=1
            else:
                cv2.putText(disp, "S: Save", (10, disp_sz-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow("Kinect Tuner", disp)
            k = cv2.waitKey(1)
            if k in [ord('q'), 27]: break
            elif k == ord('s'):
                if save_config(cam): msg_timer = 30
    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
