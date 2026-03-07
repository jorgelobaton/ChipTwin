# Optionally do the shape completion for the object points (including both suface and interior points)
# Do the volume sampling for the object points, prioritize the original object points, then surface points, then interior points

import numpy as np
import open3d as o3d
import pickle
import os
import matplotlib.pyplot as plt
import trimesh
import cv2
from utils.align_util import as_mesh
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--shape_prior", action="store_true", default=False)
parser.add_argument("--num_surface_points", type=int, default=1024)
parser.add_argument("--volume_sample_size", type=float, default=0.002)
parser.add_argument("--tool_mesh", type=str, default=None,
                    help="Path to watertight tool mesh (PLY/GLB). Points inside this volume "
                         "will be removed from the final object point cloud.")
parser.add_argument("--tool_scale", type=float, default=1.0,
                    help="Uniform scale factor applied to the tool mesh around its centroid "
                         "before subtraction. Use >1 to enlarge the cut-out (e.g. 2.0 = 2x).")
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name

# Used to judge if using the shape prior
SHAPE_PRIOR = args.shape_prior
TOOL_MESH = args.tool_mesh
TOOL_SCALE = args.tool_scale
num_surface_points = args.num_surface_points
volume_sample_size = args.volume_sample_size


# ── Tool-mesh volume subtraction helper ──────────────────────────────
def load_tool_mesh(path, scale=1.0):
    """Load a watertight tool mesh, optionally scale it about its centroid."""
    tmesh = trimesh.load(path, force="mesh")
    if not tmesh.is_watertight:
        print(f"[WARN] Tool mesh at {path} is not watertight – subtraction may be imprecise.")
    if scale != 1.0:
        centroid = tmesh.centroid.copy()
        tmesh.vertices -= centroid
        tmesh.vertices *= scale
        tmesh.vertices += centroid
        print(f"[TOOL] Scaled tool mesh by {scale}x around centroid {centroid}")
    print(f"[TOOL] Loaded tool mesh: {len(tmesh.vertices)} verts, "
          f"{len(tmesh.faces)} faces, watertight={tmesh.is_watertight}, "
          f"extents={tmesh.extents}")
    return tmesh


def subtract_tool_volume(points, tool_tmesh, colors=None):
    """
    Remove points that fall *inside* the tool mesh volume.
    Returns (filtered_points, filtered_colors, keep_mask).
    """
    inside = tool_tmesh.contains(points)
    keep = ~inside
    n_removed = inside.sum()
    print(f"[TOOL] Subtracted {n_removed}/{len(points)} points inside tool volume "
          f"({n_removed/len(points)*100:.1f}%)")
    if colors is not None:
        return points[keep], colors[keep], keep
    return points[keep], None, keep


def getSphereMesh(center, radius=0.1, color=[0, 0, 0]):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius).translate(center)
    sphere.paint_uniform_color(color)
    return sphere


def create_turntable_video(geometries, output_path, n_frames=360):
    """Render a turntable mp4 for the given Open3D geometries.
    Matches the style used in shape_prior_pcd.py."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    height, width, _ = dummy_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video_writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    for geom in geometries:
        vis.add_geometry(geom)

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


def export_subtracted_visualizations(object_pts, surface_pts, interior_pts,
                                     tool_tmesh, output_dir):
    """
    Generate visualization videos for tool-subtracted results, matching the
    style of shape_prior_pcd.py (shell=red, surface=green, interior=blue,
    mesh=light purple).

    Also boolean-subtracts the tool from the original alpha-shape mesh and
    exports the result as GLB + PLY.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── 1. Build coloured point clouds ──────────────────────────────
    shell_pcd = o3d.geometry.PointCloud()
    shell_pcd.points = o3d.utility.Vector3dVector(object_pts)
    shell_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # red – observed shell

    surface_pcd = o3d.geometry.PointCloud()
    if len(surface_pts) > 0:
        surface_pcd.points = o3d.utility.Vector3dVector(surface_pts)
        surface_pcd.paint_uniform_color([0.0, 1.0, 0.4])  # green – surface fill

    interior_pcd = o3d.geometry.PointCloud()
    if len(interior_pts) > 0:
        interior_pcd.points = o3d.utility.Vector3dVector(interior_pts)
        interior_pcd.paint_uniform_color([0.0, 0.5, 1.0])  # blue – interior fill

    # ── 2. Boolean-subtract tool from original alpha-shape mesh ─────
    original_mesh_path = os.path.join(output_dir, "final_mesh.glb")
    subtracted_mesh_vis = None
    if os.path.exists(original_mesh_path):
        try:
            original_tmesh = trimesh.load(original_mesh_path, force="mesh")
            subtracted_tmesh = trimesh.boolean.difference(
                [original_tmesh, tool_tmesh], engine="manifold"
            )
            if subtracted_tmesh is not None and len(subtracted_tmesh.faces) > 0:
                # Export the boolean-subtracted mesh
                glb_out = os.path.join(output_dir, "final_mesh_tool_subtracted.glb")
                subtracted_tmesh.export(glb_out)
                print(f"[TOOL] Exported boolean-subtracted mesh to {glb_out}")

                ply_out = os.path.join(output_dir, "final_mesh_tool_subtracted.ply")
                subtracted_tmesh.export(ply_out)
                print(f"[TOOL] Exported boolean-subtracted mesh to {ply_out}")

                # Convert to Open3D for visualization
                subtracted_mesh_vis = o3d.geometry.TriangleMesh()
                subtracted_mesh_vis.vertices = o3d.utility.Vector3dVector(
                    subtracted_tmesh.vertices
                )
                subtracted_mesh_vis.triangles = o3d.utility.Vector3iVector(
                    subtracted_tmesh.faces
                )
                subtracted_mesh_vis.compute_vertex_normals()
                subtracted_mesh_vis.paint_uniform_color([0.7, 0.7, 0.9])  # light purple
            else:
                print("[WARN] Boolean subtraction produced empty mesh, falling back to point cloud only.")
        except Exception as e:
            print(f"[WARN] Boolean subtraction failed ({e}), falling back to point cloud only.")
    else:
        print(f"[WARN] Original mesh not found at {original_mesh_path}, skipping mesh subtraction.")

    # If boolean subtraction failed or was not available, build a simple mesh
    # from the subtracted points (Poisson reconstruction) as fallback.
    if subtracted_mesh_vis is None:
        all_pts = np.concatenate(
            [p for p in [object_pts, surface_pts, interior_pts] if len(p) > 0], axis=0
        )
        if len(all_pts) > 100:
            pcd_temp = o3d.geometry.PointCloud()
            pcd_temp.points = o3d.utility.Vector3dVector(all_pts)
            pcd_temp.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
            )
            mesh_fb, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd_temp, depth=8, linear_fit=True
            )
            densities = np.asarray(densities)
            mesh_fb.remove_vertices_by_mask(densities < np.quantile(densities, 0.05))
            mesh_fb.compute_vertex_normals()
            mesh_fb.paint_uniform_color([0.7, 0.7, 0.9])
            subtracted_mesh_vis = mesh_fb

            ply_out = os.path.join(output_dir, "final_mesh_tool_subtracted.ply")
            o3d.io.write_triangle_mesh(ply_out, mesh_fb)
            print(f"[TOOL] Exported fallback Poisson mesh to {ply_out}")

    # ── 3. Render videos (matching shape_prior_pcd.py layout) ───────
    geom_lists = []

    # Build tool mesh visualization (semi-transparent wireframe)
    tool_vis = o3d.geometry.TriangleMesh()
    tool_vis.vertices = o3d.utility.Vector3dVector(tool_tmesh.vertices)
    tool_vis.triangles = o3d.utility.Vector3iVector(tool_tmesh.faces)
    tool_vis.compute_vertex_normals()
    tool_vis.paint_uniform_color([1.0, 0.4, 0.4])  # salmon – tool overlay

    # Video 1: Shell only (observed object points after subtraction)
    v1 = os.path.join(output_dir, "vis_tool_sub_shell_only.mp4")
    create_turntable_video([shell_pcd], v1)
    print(f"  [1/5] Shell only (subtracted) -> {v1}")

    # Video 2: Shell + fill (shell + surface + interior, no mesh)
    pcds_fill = [shell_pcd]
    if len(surface_pts) > 0:
        pcds_fill.append(surface_pcd)
    if len(interior_pts) > 0:
        pcds_fill.append(interior_pcd)
    v2 = os.path.join(output_dir, "vis_tool_sub_shell_with_fill.mp4")
    create_turntable_video(pcds_fill, v2)
    print(f"  [2/5] Shell + fill (subtracted) -> {v2}")

    # Video 3: Mesh only (boolean-subtracted or Poisson fallback)
    if subtracted_mesh_vis is not None:
        v3 = os.path.join(output_dir, "vis_tool_sub_mesh_only.mp4")
        create_turntable_video([subtracted_mesh_vis], v3)
        print(f"  [3/5] Mesh only (subtracted) -> {v3}")
    else:
        print(f"  [3/5] SKIPPED (no mesh available)")

    # Video 4: Mesh + shell overlay
    if subtracted_mesh_vis is not None:
        v4 = os.path.join(output_dir, "vis_tool_sub_mesh_with_shell.mp4")
        create_turntable_video([subtracted_mesh_vis, shell_pcd], v4)
        print(f"  [4/5] Mesh + shell overlay (subtracted) -> {v4}")
    else:
        print(f"  [4/5] SKIPPED (no mesh available)")

    # Video 5: Combined (legacy-style) – mesh + all point clouds
    combined = list(pcds_fill)
    if subtracted_mesh_vis is not None:
        combined.append(subtracted_mesh_vis)
    v5 = os.path.join(output_dir, "final_mesh_tool_subtracted.mp4")
    create_turntable_video(combined, v5)
    print(f"  [5/5] Combined -> {v5}")


def process_unique_points(track_data):
    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions_valid = track_data["object_motions_valid"]
    controller_points = track_data["controller_points"]

    # Get the unique index in the object points
    first_object_points = object_points[0]
    unique_idx = np.unique(first_object_points, axis=0, return_index=True)[1]
    object_points = object_points[:, unique_idx, :]
    object_colors = object_colors[:, unique_idx, :]
    object_visibilities = object_visibilities[:, unique_idx]
    object_motions_valid = object_motions_valid[:, unique_idx]

    # Make sure all points are above the ground
    # object_points[object_points[..., 2] > 0, 2] = 0

    if SHAPE_PRIOR:
        shape_mesh_path = f"{base_path}/{case_name}/shape/matching/final_mesh.glb"
        trimesh_mesh = trimesh.load(shape_mesh_path, force="mesh")
        trimesh_mesh = as_mesh(trimesh_mesh)
        # Sample the surface points
        surface_points, _ = trimesh.sample.sample_surface(
            trimesh_mesh, num_surface_points
        )
        # Sample the interior points
        interior_points = trimesh.sample.volume_mesh(trimesh_mesh, 10000)

    if SHAPE_PRIOR:
        all_points = np.concatenate(
            [surface_points, interior_points, object_points[0]], axis=0
        )
    else:
        all_points = object_points[0]
    # Do the volume sampling for the object points, prioritize the original object points, then surface points, then interior points
    min_bound = np.min(all_points, axis=0)
    index = []
    grid_flag = {}
    for i in range(object_points.shape[1]):
        grid_index = tuple(
            np.floor((object_points[0, i] - min_bound) / volume_sample_size).astype(int)
        )
        if grid_index not in grid_flag:
            grid_flag[grid_index] = 1
            index.append(i)
    if SHAPE_PRIOR:
        final_surface_points = []
        for i in range(surface_points.shape[0]):
            grid_index = tuple(
                np.floor((surface_points[i] - min_bound) / volume_sample_size).astype(
                    int
                )
            )
            if grid_index not in grid_flag:
                grid_flag[grid_index] = 1
                final_surface_points.append(surface_points[i])
        final_interior_points = []
        for i in range(interior_points.shape[0]):
            grid_index = tuple(
                np.floor((interior_points[i] - min_bound) / volume_sample_size).astype(
                    int
                )
            )
            if grid_index not in grid_flag:
                grid_flag[grid_index] = 1
                final_interior_points.append(interior_points[i])
        all_points = np.concatenate(
            [final_surface_points, final_interior_points, object_points[0][index]],
            axis=0,
        )
    else:
        all_points = object_points[0][index]

    # ── Tool Volume Subtraction ──────────────────────────────────────
    # Remove any chip points (and shape-prior points) that fall inside the
    # tool mesh.  This ensures the final reconstruction only contains the
    # metal chip, not the cutting tool that was captured together with it.
    if TOOL_MESH is not None:
        tool_tmesh = load_tool_mesh(TOOL_MESH, scale=TOOL_SCALE)

        # Filter the object-point index list (removes tracked points inside tool)
        obj_pts_frame0 = object_points[0, index]
        inside_obj = tool_tmesh.contains(obj_pts_frame0)
        new_index = [idx for idx, is_in in zip(index, inside_obj) if not is_in]
        print(f"[TOOL] Object points: kept {len(new_index)}/{len(index)} "
              f"(removed {len(index)-len(new_index)} inside tool)")
        index = new_index

        if SHAPE_PRIOR:
            # Filter surface points
            if len(final_surface_points) > 0:
                sp = np.array(final_surface_points)
                keep_sp = ~tool_tmesh.contains(sp)
                final_surface_points = sp[keep_sp].tolist()
                print(f"[TOOL] Surface points: kept {keep_sp.sum()}/{len(keep_sp)}")
            # Filter interior points
            if len(final_interior_points) > 0:
                ip = np.array(final_interior_points)
                keep_ip = ~tool_tmesh.contains(ip)
                final_interior_points = ip[keep_ip].tolist()
                print(f"[TOOL] Interior points: kept {keep_ip.sum()}/{len(keep_ip)}")

        # Rebuild all_points after subtraction
        if SHAPE_PRIOR:
            all_points = np.concatenate(
                [final_surface_points, final_interior_points, object_points[0][index]],
                axis=0,
            )
        else:
            all_points = object_points[0][index]

        print(f"[TOOL] Final point count after tool subtraction: {all_points.shape[0]}")

        # Generate visualization videos (matching shape_prior_pcd.py style)
        vis_output_dir = f"{base_path}/{case_name}/shape/matching"
        export_subtracted_visualizations(
            object_pts=object_points[0][index],
            surface_pts=np.array(final_surface_points) if SHAPE_PRIOR and len(final_surface_points) > 0 else np.zeros((0, 3)),
            interior_pts=np.array(final_interior_points) if SHAPE_PRIOR and len(final_interior_points) > 0 else np.zeros((0, 3)),
            tool_tmesh=tool_tmesh,
            output_dir=vis_output_dir,
        )

    # Render the final pcd with interior filling as a turntable video
    all_pcd = o3d.geometry.PointCloud()
    all_pcd.points = o3d.utility.Vector3dVector(all_points)
    coorindate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    height, width, _ = dummy_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video_writer = cv2.VideoWriter(
        f"{base_path}/{case_name}/final_pcd.mp4", fourcc, 30, (width, height)
    )

    vis.add_geometry(all_pcd)
    # vis.add_geometry(coorindate)
    view_control = vis.get_view_control()
    for j in range(360):
        view_control.rotate(10, 0)
        vis.poll_events()
        vis.update_renderer()
        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    vis.destroy_window()

    track_data.pop("object_points")
    track_data.pop("object_colors")
    track_data.pop("object_visibilities")
    track_data.pop("object_motions_valid")
    track_data["object_points"] = object_points[:, index, :]
    track_data["object_colors"] = object_colors[:, index, :]
    track_data["object_visibilities"] = object_visibilities[:, index]
    track_data["object_motions_valid"] = object_motions_valid[:, index]
    if SHAPE_PRIOR:
        track_data["surface_points"] = np.array(final_surface_points)
        track_data["interior_points"] = np.array(final_interior_points)
    else:
        track_data["surface_points"] = np.zeros((0, 3))
        track_data["interior_points"] = np.zeros((0, 3))

    return track_data


def visualize_track(track_data):
    object_points = track_data["object_points"]
    object_colors = track_data["object_colors"]
    object_visibilities = track_data["object_visibilities"]
    object_motions_valid = track_data["object_motions_valid"]
    controller_points = track_data["controller_points"]

    frame_num = object_points.shape[0]

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    dummy_frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    height, width, _ = dummy_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video_writer = cv2.VideoWriter(
        f"{base_path}/{case_name}/final_data.mp4", fourcc, 30, (width, height)
    )

    controller_meshes = []
    prev_center = []

    y_min, y_max = np.min(object_points[0, :, 1]), np.max(object_points[0, :, 1])
    y_normalized = (object_points[0, :, 1] - y_min) / (y_max - y_min)
    rainbow_colors = plt.cm.rainbow(y_normalized)[:, :3]

    for i in range(frame_num):
        object_pcd = o3d.geometry.PointCloud()
        object_pcd.points = o3d.utility.Vector3dVector(
            object_points[i, np.where(object_visibilities[i])[0], :]
        )
        # object_pcd.colors = o3d.utility.Vector3dVector(
        #     object_colors[i, np.where(object_motions_valid[i])[0], :]
        # )
        object_pcd.colors = o3d.utility.Vector3dVector(
            rainbow_colors[np.where(object_visibilities[i])[0]]
        )

        if i == 0:
            render_object_pcd = object_pcd
            vis.add_geometry(render_object_pcd)
            # Use sphere mesh for each controller point
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                origin_color = [1, 0, 0]
                controller_meshes.append(
                    getSphereMesh(origin, color=origin_color, radius=0.01)
                )
                vis.add_geometry(controller_meshes[-1])
                prev_center.append(origin)
            # Adjust the viewpoint to center on the object
            bbox = object_pcd.get_axis_aligned_bounding_box()
            center = bbox.get_center()
            view_control = vis.get_view_control()
            view_control.set_lookat(center)
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(0.8)
        else:
            render_object_pcd.points = o3d.utility.Vector3dVector(object_pcd.points)
            render_object_pcd.colors = o3d.utility.Vector3dVector(object_pcd.colors)
            vis.update_geometry(render_object_pcd)
            for j in range(controller_points.shape[1]):
                origin = controller_points[i, j]
                controller_meshes[j].translate(origin - prev_center[j])
                vis.update_geometry(controller_meshes[j])
                prev_center[j] = origin
        vis.poll_events()
        vis.update_renderer()

        frame = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        frame = (frame * 255).astype(np.uint8)
        # Convert RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)


if __name__ == "__main__":
    with open(f"{base_path}/{case_name}/track_process_data.pkl", "rb") as f:
        track_data = pickle.load(f)

    track_data = process_unique_points(track_data)

    with open(f"{base_path}/{case_name}/final_data.pkl", "wb") as f:
        pickle.dump(track_data, f)

    visualize_track(track_data)
