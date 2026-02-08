import numpy as np
import open3d as o3d
import argparse

def main(npy_path):

    # 1. Load and prepare data
    depth_map = np.load(npy_path)
    h, w = depth_map.shape

    # Generate 3D points: (x, y, z)
    # Flattening the grid to create a (N, 3) array
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    points = np.stack((x.flatten(), y.flatten(), depth_map.flatten()), axis=-1)

    # 2. Create Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # 3. Optional: Estimate normals for better lighting
    pcd.estimate_normals()

    # 4. Visualize
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--npy_path', type=str, required=True, help='Path to the .npy depth map file')
    args = argparser.parse_args()
    main(args.npy_path)