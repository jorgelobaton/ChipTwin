import cv2
import numpy as np
import open3d as o3d
import pickle
import os
import time
from datetime import datetime

# Import Camera Classes
from cams.camera_d405 import D405Camera
from cams.camera_kinect import KinectCamera

def load_calibration(filename="calibrate.pkl"):
    if not os.path.exists(filename):
        print(f"[WARN] Calibration file {filename} not found! Merged PCD will be unaligned.")
        return None
    with open(filename, "rb") as f:
        return pickle.load(f)

def run():
    print("[INFO] Starting Multi-Camera Recorder...")

    # 1. Discover & Initialize Cameras
    cameras = {}
    
    # Init D405s
    import pyrealsense2 as rs
    ctx = rs.context()
    for dev in ctx.query_devices():
        if "D405" in dev.get_info(rs.camera_info.name):
            serial = dev.get_info(rs.camera_info.serial_number)
            try:
                cid = f"D405_{serial}"
                cameras[cid] = D405Camera(serial_number=serial)
                print(f"[INIT] {cid}")
            except Exception as e: print(f"[ERR] {cid}: {e}")

    # Init Kinects
    from pyk4a import connected_device_count
    for idx in range(connected_device_count()):
        try:
            cid = f"Kinect_{idx}"
            cameras[cid] = KinectCamera(device_index=idx)
            print(f"[INIT] {cid}")
        except Exception as e: print(f"[ERR] {cid}: {e}")

    if not cameras:
        print("[FAIL] No cameras found.")
        return

    # 2. Load Calibration
    calib_data = load_calibration()
    transforms = {}
    if calib_data:
        if isinstance(calib_data, list):
            # New format: List of c2ws matches sorted camera IDs
            sorted_cids = sorted(cameras.keys())
            if len(calib_data) == len(sorted_cids):
                for i, cid in enumerate(sorted_cids):
                    transforms[cid] = calib_data[i]
                print(f"[INFO] Loaded calibration (list format) for: {list(transforms.keys())}")
            else:
                print(f"[WARN] Calibration count ({len(calib_data)}) does not match found cameras ({len(sorted_cids)}). Mismatch possible.")
        else:
            transforms = calib_data.get("c2ws_by_id", {})
            print(f"[INFO] Loaded calibration (dict format) for: {list(transforms.keys())}")

    cv2.namedWindow('Multi-Cam Recorder', cv2.WINDOW_AUTOSIZE)
    is_rec = False
    
    try:
        while True:
            # A. Synchronized Capture
            frames = {}
            for cid, cam in cameras.items():
                d, c = cam.get_frame()
                if c is not None: frames[cid] = (d, c)
            
            if len(frames) != len(cameras):
                continue

            # B. Visualization
            display_imgs = []
            for cid in sorted(frames.keys()):
                d, c = frames[cid]
                vis_scale = cameras[cid].config_data.get('visual_scale', 5)
                d_cm = cv2.applyColorMap(cv2.convertScaleAbs(d, alpha=vis_scale/100.0), cv2.COLORMAP_JET)
                d_cm[d==0] = 0
                ov = cv2.addWeighted(c, 0.6, d_cm, 0.4, 0)
                cv2.putText(ov, cid, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                display_imgs.append(cv2.resize(ov, (400, 400)))
            
            if display_imgs:
                stack = np.hstack(display_imgs)
                status_bar = np.zeros((60, stack.shape[1], 3), dtype=np.uint8)
                txt = "REC: Space (Toggle)" if is_rec else "Standby (Space to Record)"
                col = (0,0,255) if is_rec else (0,255,0)
                cv2.putText(status_bar, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
                final_vis = np.vstack((stack, status_bar))
                cv2.imshow('Multi-Cam Recorder', final_vis)

            # C. Input Handling
            k = cv2.waitKey(1)
            if k in [ord('q'), 27]: break
            elif k == ord(' '):
                print("Capturing...")
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                merged_pcd = o3d.geometry.PointCloud()
                saved_count = 0
                
                for cid, (depth, color) in frames.items():
                    cam = cameras[cid]
                    p = cam.config_data
                    
                    if hasattr(cam, 'get_intrinsics'):
                        intr = cam.get_intrinsics()
                    else:
                        intr = o3d.camera.PinholeCameraIntrinsic(720, 720, 380, 380, 360, 360)

                    # --- FIX: ENSURE CONTIGUOUS MEMORY FOR OPEN3D ---
                    depth = np.ascontiguousarray(depth)
                    color = np.ascontiguousarray(color)
                    # ------------------------------------------------

                    img_d = o3d.geometry.Image(depth)
                    img_c = o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
                    
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        img_c, img_d, 
                        depth_scale=1.0/cam.scale, 
                        depth_trunc=p['max_dist'], 
                        convert_rgb_to_intensity=False
                    )
                    
                    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
                    
                    if cid in transforms:
                        T = transforms[cid]
                        pcd.transform(T)
                    
                    indiv_fn = f"capture_{ts}_{cid}.pcd"
                    o3d.io.write_point_cloud(indiv_fn, pcd)
                    print(f"   Saved {indiv_fn}")
                    
                    merged_pcd += pcd
                    saved_count += 1

                if saved_count > 0:
                    merged_fn = f"capture_{ts}_merged.pcd"
                    o3d.io.write_point_cloud(merged_fn, merged_pcd)
                    print(f"[SUCCESS] Saved {merged_fn} ({len(merged_pcd.points)} pts)")
                    print("Showing Merged Result...")
                    o3d.visualization.draw_geometries([merged_pcd], window_name="Merged Result")
                    
                    # Also show individual PCDs sequentially
                    sorted_cids = sorted(frames.keys())
                    for cid in sorted_cids:
                        cam_fn = f"capture_{ts}_{cid}.pcd"
                        if os.path.exists(cam_fn):
                            print(f"Showing {cid} ({cam_fn})...")
                            pcd_single = o3d.io.read_point_cloud(cam_fn)
                            o3d.visualization.draw_geometries([pcd_single], window_name=f"Result: {cid}")

    finally:
        for cam in cameras.values():
            cam.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
