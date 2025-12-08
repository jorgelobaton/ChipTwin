import pyrealsense2 as rs
import sys

# Try to import pyk4a for Azure Kinect
try:
    from pyk4a import PyK4A, connected_device_count, K4AException
    AZURE_KINECT_AVAILABLE = True
except ImportError:
    AZURE_KINECT_AVAILABLE = False

def check_realsense():
    print("--- Checking Intel RealSense Devices ---")
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("No RealSense devices found.")
        return

    print(f"Found {len(devices)} RealSense devices.")
    for i, dev in enumerate(devices):
        print(f"\nDevice {i}:")
        try:
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            product_line = dev.get_info(rs.camera_info.product_line)
            print(f"  Name: {name}")
            print(f"  Serial: {serial}")
            print(f"  Product Line: {product_line}")
            
            if product_line == 'D400':
                print("  -> Compatible (Product Line is 'D400')")
            else:
                print(f"  -> WARNING: Expects 'D400', found '{product_line}'")
        except Exception as e:
            print(f"  -> Error reading device info: {e}")

def check_azure_kinect():
    print("\n--- Checking Azure Kinect Devices ---")
    
    if not AZURE_KINECT_AVAILABLE:
        print("WARNING: 'pyk4a' library not installed.")
        print("To check Azure Kinect, run: pip install pyk4a")
        return

    # Check number of connected devices
    try:
        count = connected_device_count()
    except Exception as e:
        print(f"Error querying Azure Kinect driver: {e}")
        print("Ensure the Azure Kinect Sensor SDK is installed and devices are powered.")
        return

    if count == 0:
        print("No Azure Kinect devices found.")
        return

    print(f"Found {count} Azure Kinect devices.")
    
    # Iterate and open each device to get details
    for i in range(count):
        print(f"\nDevice {i}:")
        try:
            # Open device by index
            device = PyK4A(device_id=i)
            device.open()
            
            print(f"  Name: Azure Kinect DK")
            print(f"  Serial: {device.serial}")
            print(f"  Product Line: Azure Kinect")
            print("  -> Compatible with Azure Kinect code")
            
            device.close()
        except K4AException as e:
            print(f"  -> Error: Could not open device (it might be in use). {e}")
        except Exception as e:
            print(f"  -> Unexpected error: {e}")

if __name__ == "__main__":
    check_realsense()
    check_azure_kinect()
