import open3d as o3d
import numpy as np
import sys
import os

def load_from_npz(filename):
    print(f"Detected NPZ format. Extracting data...")
    try:
        data = np.load(filename)
        points = data['points']  # Shape: (Num_Cams, H, W, 3)
        colors = data['colors']  # Shape: (Num_Cams, H, W, 3)
        masks = data['masks']    # Shape: (Num_Cams, H, W)
    except KeyError as e:
        print(f"Error: NPZ file missing key {e}. Expected 'points', 'colors', 'masks'.")
        return None

    all_pts = []
    all_cols = []

    # Iterate over each camera in the stack
    num_cams = points.shape[0]
    for i in range(num_cams):
        # Extract valid points using the mask
        mask = masks[i]
        
        # Flatten and filter
        valid_pts = points[i][mask]
        valid_cols = colors[i][mask]
        
        all_pts.append(valid_pts)
        all_cols.append(valid_cols)

    # Stack all cameras together
    if len(all_pts) > 0:
        final_pts = np.vstack(all_pts)
        final_cols = np.vstack(all_cols)
    else:
        return o3d.geometry.PointCloud()

    # Create Open3D object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(final_pts)
    pcd.colors = o3d.utility.Vector3dVector(final_cols)
    
    return pcd

def main():
    # Usage: python view_pcd.py filename.npz
    filename = sys.argv[1] if len(sys.argv) > 1 else "data/test_case/pcd/0.npz"

    if not os.path.exists(filename):
        print(f"Error: File {filename} not found.")
        return

    print(f"Loading {filename}...")

    # Load based on extension
    if filename.endswith('.npz'):
        pcd = load_from_npz(filename)
    else:
        # Standard PCD/PLY load
        pcd = o3d.io.read_point_cloud(filename)
    
    if pcd is None or pcd.is_empty():
        print("Error: Point cloud is empty or failed to load.")
        return

    print(f"Loaded {len(pcd.points)} points.")
    
    # Visualization
    # Add a coordinate frame (Red=X, Green=Y, Blue=Z)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    
    # Optional: Reset view logic can be added here if needed
    o3d.visualization.draw_geometries(
        [pcd, axes], 
        window_name=f"Viewer - {os.path.basename(filename)}",
        width=1024,
        height=768
    )

if __name__ == "__main__":
    main()
