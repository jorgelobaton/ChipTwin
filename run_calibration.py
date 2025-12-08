import pyrealsense2 as rs
from pyk4a import connected_device_count
import cv2

# Import your CameraSystem class
try:
    from qqtt.env.camera.camera_system import CameraSystem
except ImportError:
    try:
        from camera_system import CameraSystem
    except ImportError:
        print("Error: Could not import CameraSystem.")
        exit(1)

def discover_cameras():
    print("--- Discovery Mode ---")
    realsense_ctx = rs.context()
    rs_devices = realsense_ctx.query_devices()
    
    standard_serials = []
    d405_serials = []
    
    for dev in rs_devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        if "D405" in name:
            d405_serials.append(serial)
        else:
            standard_serials.append(serial)
            
    kinect_count = connected_device_count()
    kinect_indices = list(range(kinect_count))
    
    return standard_serials, d405_serials, kinect_indices

def main():
    rs_serials, d405_serials, k_indices = discover_cameras()
    
    # Initialize System with HIGH exposure for clearer board detection if needed,
    # or default (33000) is usually fine.
    cam_sys = CameraSystem(
        realsense_config={"WH": [848, 480], "fps": 30, "serials": rs_serials},
        d405_config={"WH": [1280, 720], "fps": 30, "serials": d405_serials},
        kinect_config={"mode": "WFOV_BINNED", "fps": 30, "indices": k_indices},
        exposure=10000
    )

    print("\nStarting Calibration...")
    print("Please hold the ChArUco board visible to ALL cameras.")
    print("The system will auto-save and exit when a valid pose is found.")
    
    # This blocks until success
    cam_sys.calibrate(visualize=True)
    
    print("Calibration finished. 'calibrate.pkl' saved.")

if __name__ == "__main__":
    main()
