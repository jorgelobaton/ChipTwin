import pyrealsense2 as rs
import numpy as np
import cv2
import json
import os

CONFIG_FILE = "config_d405.json"
DISPLAY_SCALE = 0.50

def load_postproc_config():
    cfg = {
        "WH": [1280, 720], "fps": 30, "preset_id": 4,
        "exposure": -1, "gain": 16, "visual_scale": 3,
        "min_dist": 0.07, "max_dist": 0.50, "depth_scale": 0.0001,
        "disparity_transform": 0, "history_fill": 1, "history_decay": 5,
        "spatial_filter": { "enable": 1, "magnitude": 2, "alpha": 0.5, "delta": 20, "holes_fill": 0 },
        "temporal_filter": { "enable": 1, "alpha": 0.4, "delta": 20, "persistence": 3 }
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                loaded = json.load(f)
                for k, v in loaded.items():
                    if isinstance(v, dict) and k in cfg: cfg[k].update(v)
                    else: cfg[k] = v
        except: pass
    return cfg

def save_config(settings):
    try:
        with open(CONFIG_FILE, 'w') as f: json.dump(settings, f, indent=4)
        return True
    except: return False

def safe_set(sensor, opt, val):
    try: sensor.set_option(opt, val)
    except: pass

def nothing(val): pass

def run():
    print(f"[INFO] Starting D405 Tuner...")
    p = load_postproc_config()
    
    # Init Pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, p['WH'][0], p['WH'][1], rs.format.bgr8, p['fps'])
    config.enable_stream(rs.stream.depth, p['WH'][0], p['WH'][1], rs.format.z16, p['fps'])
    
    prof = pipeline.start(config)
    ds = prof.get_device().first_depth_sensor()

    # Hardware Init
    safe_set(ds, rs.option.visual_preset, p['preset_id'])
    
    # Get Scale
    hw_scale = ds.get_depth_scale()
    
    # Filters
    align = rs.align(rs.stream.color)
    d2d = rs.disparity_transform(True)
    d2depth = rs.disparity_transform(False)
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()

    cv2.namedWindow('D405 Tuner', cv2.WINDOW_AUTOSIZE)

    # --- SLIDERS ---
    # Exposure
    is_auto = 1 if p['exposure'] == -1 else 0
    exp_v = 5000 if p['exposure'] == -1 else p['exposure']
    cv2.createTrackbar('Auto Exp', 'D405 Tuner', is_auto, 1, nothing)
    cv2.createTrackbar('Exposure', 'D405 Tuner', int(exp_v), 33000, nothing)
    cv2.createTrackbar('Gain', 'D405 Tuner', int(p['gain']), 248, nothing)
    
    # Range (CM)
    cv2.createTrackbar('Min Dist(cm)', 'D405 Tuner', int(p['min_dist']*100), 100, nothing)
    cv2.createTrackbar('Max Dist(cm)', 'D405 Tuner', int(p['max_dist']*100), 200, nothing)

    # Vis
    cv2.createTrackbar('Vis Scale', 'D405 Tuner', int(p['visual_scale']), 50, nothing)

    # Filters
    cv2.createTrackbar('USE DISPARITY', 'D405 Tuner', int(p['disparity_transform']), 1, nothing)
    s = p['spatial_filter']
    cv2.createTrackbar('S Enable', 'D405 Tuner', int(s['enable']), 1, nothing)
    cv2.createTrackbar('S Mag', 'D405 Tuner', int(s['magnitude']), 5, nothing)
    cv2.createTrackbar('S Alpha', 'D405 Tuner', int(s['alpha']*100), 100, nothing)
    cv2.createTrackbar('S Delta', 'D405 Tuner', int(s['delta']), 100, nothing)
    
    t = p['temporal_filter']
    cv2.createTrackbar('T Enable', 'D405 Tuner', int(t['enable']), 1, nothing)
    cv2.createTrackbar('T Alpha', 'D405 Tuner', int(t['alpha']*100), 100, nothing)
    cv2.createTrackbar('T Delta', 'D405 Tuner', int(t['delta']), 100, nothing)
    cv2.createTrackbar('T Persist', 'D405 Tuner', int(t['persistence']), 8, nothing)

    cv2.createTrackbar('HIST FILL', 'D405 Tuner', int(p['history_fill']), 1, nothing)
    cv2.createTrackbar('HIST DECAY', 'D405 Tuner', int(p['history_decay']), 200, nothing)

    hist_buf, age_buf, bg_buf, prev_st = None, None, None, None
    msg_timer = 0

    try:
        while True:
            # Read UI
            auto = cv2.getTrackbarPos('Auto Exp', 'D405 Tuner')
            exp = max(1, cv2.getTrackbarPos('Exposure', 'D405 Tuner'))
            gain = max(16, cv2.getTrackbarPos('Gain', 'D405 Tuner'))
            
            min_cm = cv2.getTrackbarPos('Min Dist(cm)', 'D405 Tuner')
            max_cm = max(min_cm+1, cv2.getTrackbarPos('Max Dist(cm)', 'D405 Tuner'))
            
            vis = max(1, cv2.getTrackbarPos('Vis Scale', 'D405 Tuner'))
            
            u_disp = cv2.getTrackbarPos('USE DISPARITY', 'D405 Tuner')
            s_en = cv2.getTrackbarPos('S Enable', 'D405 Tuner')
            s_mag = max(1, cv2.getTrackbarPos('S Mag', 'D405 Tuner'))
            s_alp = max(25, cv2.getTrackbarPos('S Alpha', 'D405 Tuner'))/100.0
            s_del = max(1, cv2.getTrackbarPos('S Delta', 'D405 Tuner'))
            
            t_en = cv2.getTrackbarPos('T Enable', 'D405 Tuner')
            t_alp = cv2.getTrackbarPos('T Alpha', 'D405 Tuner')/100.0
            t_del = max(1, cv2.getTrackbarPos('T Delta', 'D405 Tuner'))
            t_per = cv2.getTrackbarPos('T Persist', 'D405 Tuner')
            
            u_hist = cv2.getTrackbarPos('HIST FILL', 'D405 Tuner')
            h_dec = cv2.getTrackbarPos('HIST DECAY', 'D405 Tuner')

            cur_st = (auto, exp, gain, u_disp, s_en, s_mag, s_alp, s_del, t_en, t_alp, t_del, t_per, u_hist, h_dec)

            if prev_st is None: prev_st = cur_st
            if cur_st != prev_st:
                if auto: safe_set(ds, rs.option.enable_auto_exposure, 1)
                else:
                    safe_set(ds, rs.option.enable_auto_exposure, 0)
                    safe_set(ds, rs.option.exposure, exp)
                    safe_set(ds, rs.option.gain, gain)
                
                hist_buf = None
                d2d = rs.disparity_transform(True)
                d2depth = rs.disparity_transform(False)
                spatial = rs.spatial_filter()
                temporal = rs.temporal_filter()
                
                if s_en:
                    spatial.set_option(rs.option.filter_magnitude, s_mag)
                    spatial.set_option(rs.option.filter_smooth_alpha, s_alp)
                    spatial.set_option(rs.option.filter_smooth_delta, s_del)
                if t_en:
                    temporal.set_option(rs.option.filter_smooth_alpha, t_alp)
                    temporal.set_option(rs.option.filter_smooth_delta, t_del)
                    temporal.set_option(rs.option.holes_fill, t_per)
                prev_st = cur_st

            # Process
            fs = pipeline.wait_for_frames()
            fs = align.process(fs)
            df = fs.get_depth_frame()
            cf = fs.get_color_frame()
            if not df or not cf: continue
            
            raw = np.asanyarray(df.get_data())
            col = np.asanyarray(cf.get_data())

            f = df
            if u_disp: f = d2d.process(f)
            if s_en: f = spatial.process(f)
            if t_en: f = temporal.process(f)
            if u_disp: f = d2depth.process(f)
            filt = np.asanyarray(f.get_data())

            if u_hist:
                if hist_buf is None:
                    hist_buf = np.zeros_like(filt)
                    age_buf = np.zeros_like(filt, dtype=np.uint16)
                    bg_buf = np.zeros_like(filt)
                val = filt > 0
                bg_buf[val] = np.maximum(bg_buf[val], filt[val])
                hist_buf[val] = filt[val]
                age_buf[val] = 0
                age_buf[~val] += 1
                if h_dec > 0:
                    dec = age_buf > h_dec
                    hist_buf[dec] = bg_buf[dec]
                    age_buf[dec] = 0
                final = hist_buf.copy()
            else:
                final = filt
                hist_buf = None

            # Apply Min/Max Visualization
            thresh_min = int((min_cm/100.0) / hw_scale)
            thresh_max = int((max_cm/100.0) / hw_scale)
            mask = (final >= thresh_min) & (final <= thresh_max)
            final[~mask] = 0

            # Display
            depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(final, alpha=vis/100.0), cv2.COLORMAP_JET)
            depth_cm[final==0] = 0
            ov = cv2.addWeighted(col, 0.6, depth_cm, 0.4, 0)
            
            w, h = int(p['WH'][0]*DISPLAY_SCALE), int(p['WH'][1]*DISPLAY_SCALE)
            disp = cv2.resize(ov, (w, h))

            info = f"Range:{min_cm}-{max_cm}cm Exp:{'Auto' if auto else exp}"
            cv2.putText(disp, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            if msg_timer > 0:
                cv2.putText(disp, "SAVED!", (w//2-40, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                msg_timer -= 1
            else:
                cv2.putText(disp, "S: Save Config", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            cv2.imshow('D405 Tuner', disp)
            
            key = cv2.waitKey(1)
            if key in [ord('q'), 27]: break
            elif key == ord('s'):
                out = {
                    "WH": p['WH'], "fps": p['fps'], "preset_id": p['preset_id'],
                    "exposure": -1 if auto else int(exp), "gain": int(gain),
                    "visual_scale": int(vis),
                    "min_dist": float(min_cm)/100.0, "max_dist": float(max_cm)/100.0,
                    "depth_scale": float(hw_scale),
                    "disparity_transform": int(u_disp),
                    "history_fill": int(u_hist), "history_decay": int(h_dec),
                    "spatial_filter": {"enable":int(s_en), "magnitude":int(s_mag), "alpha":float(s_alp), "delta":int(s_del), "holes_fill":0},
                    "temporal_filter": {"enable":int(t_en), "alpha":float(t_alp), "delta":int(t_del), "persistence":int(t_per)}
                }
                if save_config(out): msg_timer = 30

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
