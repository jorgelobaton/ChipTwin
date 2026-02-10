"""
Shape Prior from Point Cloud Shell
===================================
Alternative to Trellis shape prior that works directly from the observed
point cloud. Creates a watertight mesh by finding the optimal alpha shape
from the shell points (masked object points), then exports it as GLB
for downstream use by data_process_sample.py.

Algorithm:
1. Load the tracked object points (frame 0 = shell).
2. Build an alpha shape, binary-searching for the smallest alpha that
   yields a watertight mesh (maximum surface detail).
3. Export the mesh as shape/matching/final_mesh.glb.

Usage:
    python data_process/shape_prior_pcd.py \
        --base_path ./data/different_types --case_name demo_53

This replaces both shape_prior.py (Trellis) AND align.py (alignment),
since the shell points are already in world coordinates.
"""

import numpy as np
import open3d as o3d
import trimesh
import pickle
import os
import cv2
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--alpha_min", type=float, default=0.005,
                    help="Minimum alpha to search (finest detail)")
parser.add_argument("--alpha_max", type=float, default=0.10,
                    help="Maximum alpha to search (coarsest)")
parser.add_argument("--alpha_steps", type=int, default=200,
                    help="Number of steps in alpha search")
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name


def find_optimal_alpha(pcd, alpha_min, alpha_max, n_steps):
    """
    Binary-style search for the smallest alpha value that produces
    a watertight mesh from the point cloud.
    Returns (mesh, alpha) or (None, None) if no watertight mesh found.
    """
    alphas = np.linspace(alpha_min, alpha_max, n_steps)
    best_mesh = None
    best_alpha = None

    for alpha in alphas:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        if mesh.is_watertight() and len(mesh.triangles) > 0:
            best_mesh = mesh
            best_alpha = alpha
            break  # First watertight = smallest alpha = most detail

    if best_mesh is None:
        # Fallback: try convex hull
        print("[WARN] No watertight alpha shape found. Falling back to convex hull.")
        best_mesh, _ = pcd.compute_convex_hull()
        best_alpha = -1.0

    return best_mesh, best_alpha


def create_turntable_video(geometries, output_path, n_frames=360):
    """Render a turntable video of the given geometries."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    height, width, _ = dummy_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video_writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    for g in geometries:
        vis.add_geometry(g)

    view_control = vis.get_view_control()
    for _ in range(n_frames):
        view_control.rotate(10, 0)
        vis.poll_events()
        vis.update_renderer()
        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)

    video_writer.release()
    vis.destroy_window()


if __name__ == "__main__":
    data_dir = f"{base_path}/{case_name}"
    output_dir = f"{data_dir}/shape/matching"
    os.makedirs(output_dir, exist_ok=True)

    # ---- 1. Load shell points from dense PCD (Frame 0) ----
    # Use dense point cloud instead of sparse tracks for high-quality shape prior
    pcd_path = f"{data_dir}/pcd/0.npz"
    mask_path = f"{data_dir}/mask/processed_masks.pkl"
    
    print(f"[INFO] Loading dense shell from {pcd_path}")
    pcd_data = np.load(pcd_path)
    points_all = pcd_data["points"]  # (N_cams, H, W, 3)
    colors_all = pcd_data["colors"]
    
    with open(mask_path, "rb") as f:
        masks_all = pickle.load(f)
        
    shell_pts_list = []
    shell_colors_list = []
    
    # Iterate over cameras (usually 2, D405s)
    n_cams = points_all.shape[0]
    for cam_idx in range(n_cams):
        if cam_idx in masks_all[0]:
            mask = masks_all[0][cam_idx]["object"]  # frame 0, camera cam_idx
            # Apply mask
            pts = points_all[cam_idx][mask]
            cols = colors_all[cam_idx][mask]
            
            # Simple validity check (not zero/nan)
            valid = np.all(pts != 0, axis=1) # Assuming 0,0,0 is background or invalid in lift
            shell_pts_list.append(pts[valid])
            shell_colors_list.append(cols[valid])
            
    shell_pts = np.concatenate(shell_pts_list, axis=0)
    shell_colors = np.concatenate(shell_colors_list, axis=0)

    # Downsample if too large (e.g. > 100k points might be slow for alpha shape)
    # 1. Voxel downsample to regularize (merge duplicates)
    pcd_temp = o3d.geometry.PointCloud()
    pcd_temp.points = o3d.utility.Vector3dVector(shell_pts)
    pcd_temp = pcd_temp.voxel_down_sample(voxel_size=0.002)
    
    # 2. Statistical outlier removal to clean noise
    pcd_temp, _ = pcd_temp.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    shell_pts = np.asarray(pcd_temp.points)
    shell_colors = np.asarray(pcd_temp.colors) if pcd_temp.colors else np.zeros((shell_pts.shape[0], 3))

    # 3. Random downsample if still too heavy for Qhull
    MAX_POINTS = 10000
    if shell_pts.shape[0] > MAX_POINTS:
        print(f"[INFO] Downsampling shell from {shell_pts.shape[0]} to {MAX_POINTS}")
        indices = np.random.choice(shell_pts.shape[0], MAX_POINTS, replace=False)
        shell_pts = shell_pts[indices]
        shell_colors = shell_colors[indices]

    print(f"[INFO] Shell points: {shell_pts.shape[0]}")
    print(f"[INFO] Bounding box: {shell_pts.min(axis=0)} -> {shell_pts.max(axis=0)}")
    print(f"[INFO] Centroid: {shell_pts.mean(axis=0)}")

    # ---- 2. Build point cloud ----
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(shell_pts)
    # Colors might be lost if we didn't preserve them in pcd_temp correctly if they weren't there
    if len(shell_colors) == len(shell_pts):
         pcd.colors = o3d.utility.Vector3dVector(shell_colors)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )

    # ---- 3. Find optimal watertight alpha shape ----
    # Reduce steps to avoid long waits, and start from a safer min alpha
    alpha_min = max(args.alpha_min, 0.01) # 0.005 might be too small
    alpha_steps = 50 
    print(f"[INFO] Searching for optimal alpha in [{alpha_min}, {args.alpha_max}] with {alpha_steps} steps...")
    vol_mesh, best_alpha = find_optimal_alpha(
        pcd, alpha_min, args.alpha_max, alpha_steps
    )
    print(f"[INFO] Best alpha: {best_alpha:.4f}")
    print(f"[INFO] Mesh: verts={len(vol_mesh.vertices)}, tris={len(vol_mesh.triangles)}, watertight={vol_mesh.is_watertight()}")

    # ---- 4. Export as GLB (compatible with data_process_sample.py) ----
    vol_mesh.compute_vertex_normals()
    
    # Interpolate vertex colors from the original pcd
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    mesh_verts = np.asarray(vol_mesh.vertices)
    mesh_colors = []
    for v in mesh_verts:
        _, idx, _ = pcd_tree.search_knn_vector_3d(v, 1)
        mesh_colors.append(np.asarray(pcd.colors)[idx[0]])
    mesh_colors = np.array(mesh_colors)
    vol_mesh.vertex_colors = o3d.utility.Vector3dVector(mesh_colors)

    verts = np.asarray(vol_mesh.vertices)
    faces = np.asarray(vol_mesh.triangles)
    # trimesh expects colors in 0-255 uint8 or 0-1 float
    tmesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=mesh_colors)

    glb_path = f"{output_dir}/final_mesh.glb"
    tmesh.export(glb_path)
    print(f"[INFO] Exported mesh to {glb_path}")

    # ---- 5. Verify: sample like data_process_sample.py would ----
    surface_pts, _ = trimesh.sample.sample_surface(tmesh, 1024)
    interior_pts = trimesh.sample.volume_mesh(tmesh, 10000)
    print(f"[INFO] Verification sampling:")
    print(f"  Surface points: {surface_pts.shape}")
    print(f"  Interior points: {interior_pts.shape}")
    print(f"  Shell (observed): {shell_pts.shape[0]}")
    print(f"  Total fill: {surface_pts.shape[0] + interior_pts.shape[0] + shell_pts.shape[0]}")

    # ---- 6. Visualization (4 turntable videos) ----
    print("[INFO] Generating visualization videos ...")

    # Shared geometries
    shell_pcd = o3d.geometry.PointCloud()
    shell_pcd.points = o3d.utility.Vector3dVector(shell_pts)
    shell_pcd.paint_uniform_color([1.0, 0.0, 0.0])

    interior_pcd = o3d.geometry.PointCloud()
    interior_pcd.points = o3d.utility.Vector3dVector(interior_pts)
    interior_pcd.paint_uniform_color([0.0, 0.5, 1.0])

    surface_pcd = o3d.geometry.PointCloud()
    surface_pcd.points = o3d.utility.Vector3dVector(surface_pts)
    surface_pcd.paint_uniform_color([0.0, 1.0, 0.4])

    vol_mesh_vis = o3d.geometry.TriangleMesh(vol_mesh)
    vol_mesh_vis.paint_uniform_color([0.7, 0.7, 0.9])
    vol_mesh_vis.compute_vertex_normals()

    # 1. Shell only (observed point cloud)
    v1 = f"{output_dir}/vis_shell_only.mp4"
    create_turntable_video([shell_pcd], v1)
    print(f"  [1/4] Shell only -> {v1}")

    # 2. Shell + fill (shell + surface + interior, no mesh)
    v2 = f"{output_dir}/vis_shell_with_fill.mp4"
    create_turntable_video([shell_pcd, surface_pcd, interior_pcd], v2)
    print(f"  [2/4] Shell + fill -> {v2}")

    # 3. Mesh only (alpha shape mesh, no points)
    v3 = f"{output_dir}/vis_mesh_only.mp4"
    create_turntable_video([vol_mesh_vis], v3)
    print(f"  [3/4] Mesh only -> {v3}")

    # 4. Mesh + shell overlay
    v4 = f"{output_dir}/vis_mesh_with_shell.mp4"
    create_turntable_video([vol_mesh_vis, shell_pcd], v4)
    print(f"  [4/4] Mesh + shell overlay -> {v4}")

    # Keep the combined one as the default (backwards compat)
    video_path = f"{output_dir}/shape_prior_pcd.mp4"
    create_turntable_video([shell_pcd, interior_pcd, vol_mesh_vis], video_path)
    print(f"  [+]   Combined (legacy) -> {video_path}")

    print("[DONE] Shape prior from point cloud complete.")
