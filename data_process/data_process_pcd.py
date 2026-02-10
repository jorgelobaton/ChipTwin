# Merge the RGB-D data from multiple cameras into a single point cloud in world coordinate
# Do some depth filtering to make the point cloud more clean

import numpy as np
import open3d as o3d
import json
import pickle
import cv2
from tqdm import tqdm
import os
from argparse import ArgumentParser
import glob

parser = ArgumentParser()
parser.add_argument(
    "--base_path",
    type=str,
    required=True,
)
parser.add_argument("--case_name", type=str, required=True)
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name


# Use code from https://github.com/Jianghanxiao/Helper3D/blob/master/open3d_RGBD/src/camera/cameraHelper.py
def getCamera(
    transformation,
    fx,
    fy,
    cx,
    cy,
    scale=1,
    coordinate=True,
    shoot=False,
    length=4,
    color=np.array([0, 1, 0]),
    z_flip=False,
):
    # Return the camera and its corresponding frustum framework
    if coordinate:
        camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        camera.transform(transformation)
    else:
        camera = o3d.geometry.TriangleMesh()
    # Add origin and four corner points in image plane
    points = []
    camera_origin = np.array([0, 0, 0, 1])
    points.append(np.dot(transformation, camera_origin)[0:3])
    # Calculate the four points for of the image plane
    magnitude = (cy**2 + cx**2 + fx**2) ** 0.5
    if z_flip:
        plane_points = [[-cx, -cy, fx], [-cx, cy, fx], [cx, -cy, fx], [cx, cy, fx]]
    else:
        plane_points = [[-cx, -cy, -fx], [-cx, cy, -fx], [cx, -cy, -fx], [cx, cy, -fx]]
    for point in plane_points:
        point = list(np.array(point) / magnitude * scale)
        temp_point = np.array(point + [1])
        points.append(np.dot(transformation, temp_point)[0:3])
    # Draw the camera framework
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 4], [1, 3], [3, 4]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    meshes = [camera, line_set]

    if shoot:
        shoot_points = []
        shoot_points.append(np.dot(transformation, camera_origin)[0:3])
        shoot_points.append(np.dot(transformation, np.array([0, 0, -length, 1]))[0:3])
        shoot_lines = [[0, 1]]
        shoot_line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(shoot_points),
            lines=o3d.utility.Vector2iVector(shoot_lines),
        )
        shoot_line_set.paint_uniform_color(color)
        meshes.append(shoot_line_set)

    return meshes


def getPcdFromDepth(depth, intrinsic):
    H, W = depth.shape
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    x = x.reshape(-1)
    y = y.reshape(-1)
    depth = depth.reshape(-1)
    points = np.stack([x, y, np.ones_like(x)], axis=1)
    points = points * depth[:, None]
    points = points @ np.linalg.inv(intrinsic).T
    points = points.reshape(H, W, 3)
    return points


def get_pcd_from_data(path, frame_idx, camera_ids, intrinsics, c2ws, caps=None, depth_scales=None):
    total_points = []
    total_colors = []
    total_masks = []
    for i, cam_id in enumerate(camera_ids):
        if caps is not None and cam_id in caps:
            ret, color = caps[cam_id].read()
            if not ret:
                print(f"Warning: Could not read frame {frame_idx} from {cam_id}")
                # Fallback to zeros if needed, or handle error
                color = np.zeros((intrinsics[i][1, 2]*2, intrinsics[i][0, 2]*2, 3), dtype=np.uint8)
        else:
            # Fallback for individual PNGs if they exist
            color = cv2.imread(f"{path}/color/{cam_id}/{frame_idx}.png")
            if color is None:
                # Fallback to video if cap was not provided but file is missing
                cap = cv2.VideoCapture(f"{path}/color/{cam_id}.mp4")
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, color = cap.read()
                cap.release()

        if color is None:
            raise FileNotFoundError(
                f"Missing color frame {frame_idx} for camera '{cam_id}'. Check {path}/color/{cam_id}/ or {path}/color/{cam_id}.mp4"
            )
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = color.astype(np.float32) / 255.0
        depth_raw = np.load(f"{path}/depth/{cam_id}/{frame_idx}.npy")
        scale = 0.001
        if depth_scales is not None and len(depth_scales) > i:
            scale = float(depth_scales[i])
        depth = depth_raw.astype(np.float32) * scale

        points = getPcdFromDepth(
            depth,
            intrinsic=intrinsics[i],
        )
        masks = np.logical_and(points[:, :, 2] > 0.05, points[:, :, 2] < 1.5)
        points_flat = points.reshape(-1, 3)
        # Transform points to world coordinates using homogeneous transformation
        homogeneous_points = np.hstack(
            (points_flat, np.ones((points_flat.shape[0], 1)))
        )
        points_world = np.dot(c2ws[i], homogeneous_points.T).T[:, :3]
        points_final = points_world.reshape(points.shape)
        total_points.append(points_final)
        total_colors.append(color)
        total_masks.append(masks)
    # pcd = o3d.geometry.PointCloud()
    # visualize_points = []
    # visualize_colors = []
    # for i in range(num_cam):
    #     visualize_points.append(
    #         total_points[i][total_masks[i]].reshape(-1, 3)
    #     )
    #     visualize_colors.append(
    #         total_colors[i][total_masks[i]].reshape(-1, 3)
    #     )
    # visualize_points = np.concatenate(visualize_points)
    # visualize_colors = np.concatenate(visualize_colors)
    # coordinates = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    # mask = np.logical_and(visualize_points[:, 2] > -0.15, visualize_points[:, 0] > -0.05)
    # mask = np.logical_and(mask, visualize_points[:, 0] < 0.4)
    # mask = np.logical_and(mask, visualize_points[:, 1] < 0.5)
    # mask = np.logical_and(mask, visualize_points[:, 1] > -0.2)
    # mask = np.logical_and(mask, visualize_points[:, 2] < 0.2)
    # visualize_points = visualize_points[mask]
    # visualize_colors = visualize_colors[mask]
        
    # pcd.points = o3d.utility.Vector3dVector(np.concatenate(visualize_points).reshape(-1, 3))
    # pcd.colors = o3d.utility.Vector3dVector(np.concatenate(visualize_colors).reshape(-1, 3))
    # o3d.visualization.draw_geometries([pcd])
    total_points = np.asarray(total_points)
    total_colors = np.asarray(total_colors)
    total_masks = np.asarray(total_masks)
    return total_points, total_colors, total_masks


def exist_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


if __name__ == "__main__":
    with open(f"{base_path}/{case_name}/metadata.json", "r") as f:
        data = json.load(f)
    intrinsics = np.array(data["intrinsics"])
    WH = data["WH"]
    depth_scales = data.get("depth_scales")
    
    # Get camera IDs (Human readable names)
    camera_ids = data.get("camera_ids", data.get("serial_numbers", []))
    original_ids = list(camera_ids)

    def _sorted_dirs(path):
        if not os.path.exists(path):
            return []
        dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        # numeric sort if possible
        if all(d.isdigit() for d in dirs):
            return sorted(dirs, key=lambda x: int(x))
        return sorted(dirs)

    # Prefer actual folder names if metadata IDs don't match
    color_dir = os.path.join(base_path, case_name, "color")
    depth_dir = os.path.join(base_path, case_name, "depth")
    color_ids = _sorted_dirs(color_dir)
    depth_ids = _sorted_dirs(depth_dir)
    actual_ids = color_ids if color_ids else depth_ids
    if actual_ids and set(camera_ids) != set(actual_ids):
        print("[WARN] Metadata camera IDs do not match folder names; using folder names instead.")
        camera_ids = actual_ids

        # Reorder intrinsics/depth_scales if we can map by original IDs
        if original_ids and len(original_ids) == len(intrinsics):
            id_to_index = {cid: idx for idx, cid in enumerate(original_ids)}
            indices = []
            for cid in camera_ids:
                if cid in id_to_index:
                    indices.append(id_to_index[cid])
                else:
                    # Try matching by suffix (e.g. D405_XXXX vs serial)
                    matches = [idx for oid, idx in id_to_index.items() if str(oid) in cid]
                    indices.append(matches[0] if matches else None)
            if any(idx is None for idx in indices):
                print("[WARN] Unable to fully map intrinsics to folder IDs; using original order.")
            else:
                intrinsics = intrinsics[indices]
                if depth_scales is not None:
                    depth_scales = [depth_scales[i] for i in indices]
    num_cam = len(camera_ids)
    
    # Infer frame count if not present in metadata
    if "frame_num" in data:
        frame_num = data["frame_num"]
    else:
        # Check first camera depth folder to count frames
        depth_files = glob.glob(f"{base_path}/{case_name}/depth/{camera_ids[0]}/*.npy")
        frame_num = len(depth_files)
        print(f"Metadata missing 'frame_num'. Inferred {frame_num} frames from camera {camera_ids[0]}.")

    print(f"Cameras: {camera_ids}")

    calib_path = f"{base_path}/{case_name}/calibrate.pkl"
    if not os.path.exists(calib_path):
        calib_path = "calibrate.pkl"
        print(f"Case-specific calibration not found. Using global {calib_path}")
        
    c2ws_data = pickle.load(open(calib_path, "rb"))
    if isinstance(c2ws_data, dict):
        if "c2ws_by_id" in c2ws_data:
            missing_ids = [cid for cid in camera_ids if cid not in c2ws_data["c2ws_by_id"]]
            if missing_ids:
                raise ValueError(f"Calibration missing camera IDs: {missing_ids}")
            c2ws = [c2ws_data["c2ws_by_id"][cid] for cid in camera_ids]
        elif "camera_ids" in c2ws_data and "c2ws" in c2ws_data:
            calib_ids = c2ws_data["camera_ids"]
            calib_c2ws = c2ws_data["c2ws"]
            id_to_c2w = {cid: calib_c2ws[idx] for idx, cid in enumerate(calib_ids)}
            missing_ids = [cid for cid in camera_ids if cid not in id_to_c2w]
            if missing_ids:
                raise ValueError(f"Calibration missing camera IDs: {missing_ids}")
            c2ws = [id_to_c2w[cid] for cid in camera_ids]
        else:
            raise ValueError("Unsupported calibration format. Expected camera_id mapping or ordered list.")
    else:
        c2ws = c2ws_data
        # If we swapped camera_ids, attempt to reorder c2ws from original ids
        if original_ids and camera_ids != original_ids and len(c2ws) == len(original_ids):
            id_to_index = {cid: idx for idx, cid in enumerate(original_ids)}
            indices = []
            for cid in camera_ids:
                if cid in id_to_index:
                    indices.append(id_to_index[cid])
                else:
                    matches = [idx for oid, idx in id_to_index.items() if str(oid) in cid]
                    indices.append(matches[0] if matches else None)
            if any(idx is None for idx in indices):
                print("[WARN] Unable to fully map c2ws to folder IDs; using original order.")
            else:
                c2ws = [c2ws[i] for i in indices]

    exist_dir(f"{base_path}/{case_name}/pcd")

    cameras = []
    # Visualize the cameras
    for i in range(num_cam):
        camera = getCamera(
            c2ws[i],
            intrinsics[i, 0, 0],
            intrinsics[i, 1, 1],
            intrinsics[i, 0, 2],
            intrinsics[i, 1, 2],
            z_flip=True,
            scale=0.2,
        )
        cameras += camera

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for camera in cameras:
        vis.add_geometry(camera)

    coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(coordinate)

    # Open video captures for sequential reading
    caps = {}
    for cam_id in camera_ids:
        video_file = f"{base_path}/{case_name}/color/{cam_id}.mp4"
        if os.path.exists(video_file):
            caps[cam_id] = cv2.VideoCapture(video_file)
            print(f"Opened video for camera {cam_id}")

    pcd = None
    for i in tqdm(range(frame_num)):
        points, colors, masks = get_pcd_from_data(
            f"{base_path}/{case_name}", i, camera_ids, intrinsics, c2ws, caps=caps, depth_scales=depth_scales
        )

        if i == 0:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                points.reshape(-1, 3)[masks.reshape(-1)]
            )
            pcd.colors = o3d.utility.Vector3dVector(
                colors.reshape(-1, 3)[masks.reshape(-1)]
            )
            vis.add_geometry(pcd)
            # Adjust the viewpoint
            view_control = vis.get_view_control()
            view_control.set_front([1, 0, -2])
            view_control.set_up([0, 0, -1])
            view_control.set_zoom(1)
        else:
            pcd.points = o3d.utility.Vector3dVector(
                points.reshape(-1, 3)[masks.reshape(-1)]
            )
            pcd.colors = o3d.utility.Vector3dVector(
                colors.reshape(-1, 3)[masks.reshape(-1)]
            )
            vis.update_geometry(pcd)

            vis.poll_events()
            vis.update_renderer()

        np.savez(
            f"{base_path}/{case_name}/pcd/{i}.npz",
            points=points,
            colors=colors,
            masks=masks,
        )

    # Close video captures
    for cap in caps.values():
        cap.release()
