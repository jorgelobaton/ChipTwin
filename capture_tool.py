"""
capture_tool.py – Capture & reconstruct a machine-tool mesh from all D405 cameras
===================================================================================

Workflow:
  1. Discover & initialise every connected D405 camera (uses camera_map.json &
     per-camera config_d405_X.json, identical logic to run_record.py).
  2. Load extrinsic calibration from calibrate.pkl (same as data pipeline).
  3. Capture a single synchronised snapshot from every camera.
  4. Run GroundingDINO + SAM2 on each colour image to get a binary mask of the
     tool (user supplies the text prompt, e.g. "drill bit." or "end mill.").
  5. Back-project masked depth into world-frame 3D point cloud.
  6. Merge multi-view clouds → statistical outlier removal → alpha-shape mesh.
  7. Save the watertight mesh as  <output_dir>/tool_mesh.ply  (and optionally
     the raw point cloud as tool_pcd.ply).

The resulting mesh is later consumed by process_data.py (--tool_mesh flag)
to subtract the tool volume from the reconstructed chip.

Usage:
    python capture_tool.py --prompt "end mill." --output_dir ./data/tool
    python capture_tool.py --prompt "drill bit." --output_dir ./data/tool --visualize

Notes:
  • Prompt MUST be lowercase and end with a period (GroundingDINO convention).
  • Requires the same calibrate.pkl that the recording pipeline produces.
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import cv2
import open3d as o3d
import trimesh
import torch
import pyrealsense2 as rs
from torchvision.ops import box_convert

# ── Fix SAM2 shadowing: the repo has a top-level sam2/ directory that
#    shadows the installed sam2 Python package.  Point Python at the
#    inner package so `import sam2` resolves correctly.
_sam2_pkg = os.path.join(os.path.dirname(__file__), "sam2")
if os.path.isdir(os.path.join(_sam2_pkg, "sam2")):
    sys.path.insert(0, _sam2_pkg)

from cams.camera_d405 import D405Camera

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

# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Capture tool mesh from D405 cameras")
parser.add_argument("--prompt", type=str, required=True,
                    help="GroundingDINO text prompt for the tool (lowercase, period-terminated, e.g. 'end mill.')")
parser.add_argument("--output_dir", type=str, default="./data/tool",
                    help="Directory to save tool_mesh.ply and intermediate files")
parser.add_argument("--calibration", type=str, default="calibrate.pkl",
                    help="Path to extrinsic calibration pickle")
parser.add_argument("--box_threshold", type=float, default=0.35)
parser.add_argument("--text_threshold", type=float, default=0.25)
parser.add_argument("--num_warmup", type=int, default=30,
                    help="Number of warm-up frames to let auto-exposure / filters settle")
parser.add_argument("--voxel_size", type=float, default=0.001,
                    help="Voxel down-sample size for the merged cloud (metres)")
parser.add_argument("--mesh_method", type=str, default="poisson",
                    choices=["poisson", "alpha", "cylinder"],
                    help="Meshing strategy: 'poisson' (best for partial views), "
                         "'alpha' (needs dense coverage), "
                         "'cylinder' (fit analytical cylinder to the points)")
parser.add_argument("--poisson_depth", type=int, default=8,
                    help="Octree depth for Poisson reconstruction (higher = finer)")
parser.add_argument("--alpha_min", type=float, default=0.005)
parser.add_argument("--alpha_max", type=float, default=0.10)
parser.add_argument("--alpha_steps", type=int, default=100)
parser.add_argument("--cylinder_segments", type=int, default=64,
                    help="Number of facets around the fitted cylinder")
parser.add_argument("--padding", type=float, default=1.05,
                    help="Multiplier to pad the fitted cylinder radius (1.0 = tight fit, 1.05 = 5%% larger, default)")
parser.add_argument("--visualize", action="store_true",
                    help="Show interactive Open3D viewer before saving")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 1. DISCOVER & INIT CAMERAS  (same logic as run_record.py / run_calibration.py)
# ──────────────────────────────────────────────────────────────────────
print("[1/7] Discovering D405 cameras …")
ctx = rs.context()
d405_serials = sorted([
    dev.get_info(rs.camera_info.serial_number)
    for dev in ctx.query_devices()
    if "D405" in dev.get_info(rs.camera_info.name)
])

if not d405_serials:
    print("[FATAL] No D405 cameras found.")
    sys.exit(1)

camera_map = {}
if os.path.exists("cams/camera_map.json"):
    with open("cams/camera_map.json", "r") as f:
        camera_map = json.load(f)

cameras = {}   # cid → D405Camera
for i, serial in enumerate(d405_serials):
    if serial in camera_map:
        conf = camera_map[serial]
        if not os.path.exists(conf) and os.path.exists(os.path.join("cams", conf)):
            conf = os.path.join("cams", conf)
    elif i == 0:
        conf = "cams/config_d405_0.json"
    elif i == 1:
        conf = "cams/config_d405_1.json"
    else:
        conf = "cams/config_d405.json"

    cid = f"D405_{serial}"
    print(f"  [INIT] {cid}  config={conf}")
    cameras[cid] = D405Camera(serial_number=serial, config_file=conf)

camera_ids = sorted(cameras.keys())
num_cam = len(camera_ids)
print(f"  → {num_cam} camera(s) ready.")

# ──────────────────────────────────────────────────────────────────────
# 2. LOAD CALIBRATION
# ──────────────────────────────────────────────────────────────────────
print(f"[2/7] Loading calibration from {args.calibration} …")
if not os.path.exists(args.calibration):
    print(f"[FATAL] Calibration file not found: {args.calibration}")
    sys.exit(1)

with open(args.calibration, "rb") as f:
    calib = pickle.load(f)

# calib may be a plain list (ordered by sorted serial) or a dict
if isinstance(calib, dict):
    if "c2ws_by_id" in calib:
        c2ws = [calib["c2ws_by_id"][cid] for cid in camera_ids]
    elif "camera_ids" in calib and "c2ws" in calib:
        id_to_c2w = {cid: calib["c2ws"][idx] for idx, cid in enumerate(calib["camera_ids"])}
        c2ws = [id_to_c2w[cid] for cid in camera_ids]
    else:
        raise ValueError("Unsupported calibration dict format.")
else:
    c2ws = list(calib)  # ordered by sorted serial (same as camera_ids)

c2ws = [np.array(c, dtype=np.float64) for c in c2ws]
print(f"  → Loaded {len(c2ws)} extrinsics.")

# ──────────────────────────────────────────────────────────────────────
# 3. WARM-UP & CAPTURE
# ──────────────────────────────────────────────────────────────────────
print(f"[3/7] Warming up ({args.num_warmup} frames) & capturing snapshot …")
for _ in range(args.num_warmup):
    for cam in cameras.values():
        cam.get_frame()

# Capture synchronised snapshot
color_imgs = {}   # cid → BGR uint8
depth_imgs = {}   # cid → uint16 raw
intrinsics = {}   # cid → 3×3
depth_scales = {} # cid → float

for cid in camera_ids:
    cam = cameras[cid]
    d, c = cam.get_frame()
    if d is None or c is None:
        print(f"[WARN] Failed to capture from {cid}")
        continue
    color_imgs[cid] = c
    depth_imgs[cid] = d
    intrinsics[cid] = cam.get_intrinsics_matrix()
    depth_scales[cid] = cam.scale

# Save captured images for debug
for cid in camera_ids:
    cv2.imwrite(os.path.join(args.output_dir, f"capture_{cid}_color.png"), color_imgs[cid])
    np.save(os.path.join(args.output_dir, f"capture_{cid}_depth.npy"), depth_imgs[cid])
print(f"  → Captured from {len(color_imgs)} camera(s). Saved to {args.output_dir}/")

# Stop cameras now — we no longer need the streams
for cam in cameras.values():
    cam.stop()

# ──────────────────────────────────────────────────────────────────────
# 4. SEGMENT WITH GROUNDING-DINO + SAM2
# ──────────────────────────────────────────────────────────────────────
print("[4/7] Running GroundingDINO + SAM2 segmentation …")

GROUNDING_DINO_CONFIG = "./data_process/groundedSAM_checkpoints/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT = "./data_process/groundedSAM_checkpoints/groundingdino_swint_ogc.pth"
SAM2_CHECKPOINT = "./data_process/groundedSAM_checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from groundingdino.util.inference import load_model, predict
from groundingdino.util.inference import load_image as gdino_load_image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Load models once
grounding_model = load_model(
    model_config_path=GROUNDING_DINO_CONFIG,
    model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
    device=DEVICE,
)
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

masks = {}  # cid → bool H×W
for cid in camera_ids:
    if cid not in color_imgs:
        continue

    # GroundingDINO expects an image file path → write temp
    tmp_path = os.path.join(args.output_dir, f"_tmp_{cid}.png")
    cv2.imwrite(tmp_path, color_imgs[cid])
    image_source, image_tensor = gdino_load_image(tmp_path)
    os.remove(tmp_path)

    # Detect boxes
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image_tensor,
        caption=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )

    if len(boxes) == 0:
        print(f"  [WARN] No detection for '{args.prompt}' in {cid}. Skipping camera.")
        continue

    # Convert boxes to pixel coords
    h, w = image_source.shape[:2]
    boxes_px = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes_px, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    # SAM2 prediction
    sam2_predictor.set_image(image_source)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        pred_masks, scores, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
    if pred_masks.ndim == 4:
        pred_masks = pred_masks.squeeze(1)

    # Merge all detected masks into one (in case multiple detections)
    combined = np.zeros((h, w), dtype=bool)
    for m in pred_masks:
        combined |= m.astype(bool)

    masks[cid] = combined

    # Save mask overlay for debugging
    overlay = color_imgs[cid].copy()
    overlay[combined] = (overlay[combined] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    cv2.imwrite(os.path.join(args.output_dir, f"mask_{cid}.png"), overlay)
    print(f"  {cid}: {len(boxes)} detection(s), mask covers {combined.sum()} px")

if not masks:
    print("[FATAL] No masks obtained from any camera. Check your prompt / thresholds.")
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
# 5. BACK-PROJECT  masked depth → world-frame 3D points
# ──────────────────────────────────────────────────────────────────────
print("[5/7] Back-projecting masked depth to 3D …")

all_points = []
all_colors = []

for idx, cid in enumerate(camera_ids):
    if cid not in masks or cid not in depth_imgs:
        continue

    depth_raw = depth_imgs[cid].astype(np.float32) * depth_scales[cid]
    K = intrinsics[cid]
    mask = masks[cid]

    # getPcdFromDepth returns (H, W, 3) in camera frame
    pts_cam = getPcdFromDepth(depth_raw, K)

    # Validity: depth > 0 AND inside mask
    valid = (depth_raw > 0) & mask
    pts_flat = pts_cam[valid]
    cols_flat = cv2.cvtColor(color_imgs[cid], cv2.COLOR_BGR2RGB)[valid].astype(np.float32) / 255.0

    # Transform to world frame
    homo = np.hstack([pts_flat, np.ones((pts_flat.shape[0], 1))])
    pts_world = (c2ws[idx] @ homo.T).T[:, :3]

    all_points.append(pts_world)
    all_colors.append(cols_flat)
    print(f"  {cid}: {pts_world.shape[0]} points")

merged_pts = np.concatenate(all_points, axis=0)
merged_cols = np.concatenate(all_colors, axis=0)
print(f"  → Total merged: {merged_pts.shape[0]} points")

# ──────────────────────────────────────────────────────────────────────
# 6. CLEAN & BUILD MESH
# ──────────────────────────────────────────────────────────────────────
print("[6/7] Cleaning point cloud & building alpha-shape mesh …")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(merged_pts)
pcd.colors = o3d.utility.Vector3dVector(merged_cols)

# Voxel down-sample
pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
print(f"  After voxel downsample ({args.voxel_size}m): {len(pcd.points)} pts")

# Statistical outlier removal – two passes, tighter on second
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1.5)
print(f"  After outlier removal pass 1 (nb=30, σ=1.5): {len(pcd.points)} pts")
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1.0)
print(f"  After outlier removal pass 2 (nb=50, σ=1.0): {len(pcd.points)} pts")

# DBSCAN clustering – keep only the largest cluster (the actual tool)
labels = np.array(pcd.cluster_dbscan(eps=0.005, min_points=10, print_progress=False))
if len(labels) > 0 and labels.max() >= 0:
    largest_label = np.bincount(labels[labels >= 0]).argmax()
    keep_mask = labels == largest_label
    pcd = pcd.select_by_index(np.where(keep_mask)[0])
    n_clusters = labels.max() + 1
    n_noise = (labels == -1).sum()
    print(f"  DBSCAN: {n_clusters} cluster(s), {n_noise} noise pts → kept largest ({keep_mask.sum()} pts)")

pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
)
pcd.orient_normals_consistent_tangent_plane(k=15)

# ── Interactive crop: pick two endpoints to define the tool region ────
# The user Shift+clicks on the TIP and the BASE of the tool portion
# they want to keep.  Everything outside that span (along the tool axis)
# is discarded.  This cleanly removes the chuck / holder.
print("\n  ╔══════════════════════════════════════════════════════════════╗")
print("  ║  INTERACTIVE CROP                                            ║")
print("  ║                                                              ║")
print("  ║  Shift + left-click  TWO points on the point cloud:          ║")
print("  ║    1st click → one end of the tool you want to KEEP          ║")
print("  ║    2nd click → the other end                                 ║")
print("  ║                                                              ║")
print("  ║  Then press  Q  to close the window.                         ║")
print("  ║                                                              ║")
print("  ║  (Press Q without picking to skip crop and use full cloud.)  ║")
print("  ╚══════════════════════════════════════════════════════════════╝\n")

vis = o3d.visualization.VisualizerWithEditing()
vis.create_window(window_name="Pick 2 endpoints (Shift+Click), then Q",
                  width=1280, height=720)
vis.add_geometry(pcd)
vis.run()
vis.destroy_window()
picked = vis.get_picked_points()

if len(picked) >= 2:
    pts_np_all = np.asarray(pcd.points)
    p1 = pts_np_all[picked[0]]
    p2 = pts_np_all[picked[1]]
    crop_axis = p2 - p1
    crop_len = np.linalg.norm(crop_axis)
    crop_axis_n = crop_axis / crop_len

    # Project all points onto the axis defined by the two picked points
    proj = (pts_np_all - p1) @ crop_axis_n
    # Keep points whose projection is between 0 and crop_len (with a small margin)
    margin = crop_len * 0.05  # 5% margin on each end
    keep = (proj >= -margin) & (proj <= crop_len + margin)
    pcd = pcd.select_by_index(np.where(keep)[0])
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)
    print(f"  ✓ Cropped to {len(pcd.points)} points (axis length={crop_len*1000:.1f}mm).")
else:
    print(f"  → No crop applied. Using full cloud ({len(pcd.points)} pts).")

# Save raw cloud
o3d.io.write_point_cloud(os.path.join(args.output_dir, "tool_pcd.ply"), pcd)

# ── Meshing strategy ─────────────────────────────────────────────────
pts_np = np.asarray(pcd.points)

if args.mesh_method == "cylinder":
    # ── Fit an analytical cylinder to the point cloud ────────────
    # 1. PCA to find the main axis
    centroid = pts_np.mean(axis=0)
    pts_centered = pts_np - centroid
    _, _, Vt = np.linalg.svd(pts_centered, full_matrices=False)
    axis = Vt[0]  # first principal component = cylinder axis

    # 2. Project points onto axis to find length
    proj = pts_centered @ axis
    half_len = (proj.max() - proj.min()) / 2.0
    center_along_axis = centroid + axis * (proj.max() + proj.min()) / 2.0

    # 3. Radial distances from axis to find radius
    radial = pts_centered - np.outer(proj, axis)
    radial_dists = np.linalg.norm(radial, axis=1)

    # With only partial angular coverage (cameras see one side of the
    # cylinder) the visible-arc centroid is shifted inward from the true
    # cylinder centre, so the simple median radial distance underestimates
    # the radius.
    #
    # Fix: slice along the axis, fit a least-squares circle to each
    # cross-section (this finds the true centre), and take the trimmed
    # mean of the per-slice fitted radii.
    v2, v3 = Vt[1], Vt[2]  # orthonormal basis for cross-section plane
    n_slices = 20
    slice_edges = np.linspace(proj.min(), proj.max(), n_slices + 1)
    fit_radii = []
    for si in range(n_slices):
        band = (proj >= slice_edges[si]) & (proj < slice_edges[si + 1])
        if band.sum() < 10:
            continue
        sl = pts_centered[band]
        u = sl @ v2
        v = sl @ v3
        # Least-squares circle: (u-a)^2 + (v-b)^2 = R^2
        # Linearise: u^2+v^2 = 2a*u + 2b*v + (R^2 - a^2 - b^2)
        A_ls = np.column_stack([2 * u, 2 * v, np.ones(len(u))])
        b_ls = u ** 2 + v ** 2
        result, _, _, _ = np.linalg.lstsq(A_ls, b_ls, rcond=None)
        a_fit, b_fit, c_fit = result
        R_sq = c_fit + a_fit ** 2 + b_fit ** 2
        if R_sq > 0:
            fit_radii.append(np.sqrt(R_sq))

    if len(fit_radii) >= 3:
        # Trimmed mean: drop the top and bottom 20% of fitted radii
        fit_arr = np.sort(fit_radii)
        lo = int(len(fit_arr) * 0.2)
        hi = max(lo + 1, int(len(fit_arr) * 0.8))
        radius = float(np.mean(fit_arr[lo:hi])) * args.padding
        raw_radius = float(np.mean(fit_arr[lo:hi]))
    elif len(fit_radii) > 0:
        radius = float(np.mean(fit_radii)) * args.padding
        raw_radius = float(np.mean(fit_radii))
    else:
        # Fallback: 85th percentile of radial distances
        radius = np.percentile(radial_dists, 85) * args.padding
        raw_radius = np.percentile(radial_dists, 85)

    print(f"  Fitted cylinder: radius={radius*1000:.2f}mm, "
          f"length={half_len*2*1000:.2f}mm, center={center_along_axis}")
    print(f"  Radial stats: min={radial_dists.min()*1000:.2f}mm, "
          f"median={np.median(radial_dists)*1000:.2f}mm, "
          f"p75={np.percentile(radial_dists, 75)*1000:.2f}mm, "
          f"p95={np.percentile(radial_dists, 95)*1000:.2f}mm, "
          f"max={radial_dists.max()*1000:.2f}mm")
    print(f"  Circle-fit radius (trimmed mean of {len(fit_radii)} slices): "
          f"{raw_radius*1000:.2f}mm "
          f"(×{args.padding} padding → {radius*1000:.2f}mm)")

    # 4. Build cylinder mesh with trimesh
    cyl = trimesh.creation.cylinder(
        radius=radius,
        height=half_len * 2,
        sections=args.cylinder_segments,
    )
    # Align: trimesh cylinder is along Z. Rotate to match our axis.
    z_axis = np.array([0, 0, 1.0])
    if not np.allclose(axis, z_axis):
        rot_axis = np.cross(z_axis, axis)
        rot_axis_norm = np.linalg.norm(rot_axis)
        if rot_axis_norm > 1e-8:
            rot_axis /= rot_axis_norm
            angle = np.arccos(np.clip(np.dot(z_axis, axis), -1, 1))
            # Rodrigues rotation
            K_mat = np.array([
                [0, -rot_axis[2], rot_axis[1]],
                [rot_axis[2], 0, -rot_axis[0]],
                [-rot_axis[1], rot_axis[0], 0],
            ])
            R = np.eye(3) + np.sin(angle) * K_mat + (1 - np.cos(angle)) * (K_mat @ K_mat)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = center_along_axis
            cyl.apply_transform(T)
        else:
            cyl.apply_translation(center_along_axis)
    else:
        cyl.apply_translation(center_along_axis)

    # Convert to Open3D mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(cyl.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(cyl.faces)
    mesh.compute_vertex_normals()
    best_alpha = -1
    print(f"  Cylinder mesh: verts={len(mesh.vertices)}, tris={len(mesh.triangles)}, watertight={mesh.is_watertight()}")

elif args.mesh_method == "poisson":
    # ── Poisson surface reconstruction (fills gaps from partial views) ──
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=args.poisson_depth, linear_fit=True
    )
    # Remove low-density vertices (extrapolated far from observed data)
    densities = np.asarray(densities)
    density_threshold = np.quantile(densities, 0.05)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)
    mesh.compute_vertex_normals()
    best_alpha = -1
    print(f"  Poisson mesh (depth={args.poisson_depth}): verts={len(mesh.vertices)}, "
          f"tris={len(mesh.triangles)}, watertight={mesh.is_watertight()}")

else:  # alpha
    def find_optimal_alpha(cloud, alpha_min, alpha_max, n_steps):
        alphas = np.linspace(alpha_min, alpha_max, n_steps)
        for alpha in alphas:
            m = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud, alpha)
            if m.is_watertight() and len(m.triangles) > 0:
                return m, alpha
        print("  [WARN] No watertight alpha shape found → using convex hull.")
        m, _ = cloud.compute_convex_hull()
        return m, -1.0

    mesh, best_alpha = find_optimal_alpha(pcd, args.alpha_min, args.alpha_max, args.alpha_steps)
    mesh.compute_vertex_normals()
    print(f"  Alpha = {best_alpha:.4f}  |  verts = {len(mesh.vertices)}  |  "
          f"tris = {len(mesh.triangles)}  |  watertight = {mesh.is_watertight()}")

# ──────────────────────────────────────────────────────────────────────
# 7. SAVE
# ──────────────────────────────────────────────────────────────────────
print("[7/7] Saving tool mesh …")

mesh_path = os.path.join(args.output_dir, "tool_mesh.ply")
o3d.io.write_triangle_mesh(mesh_path, mesh)
print(f"  → {mesh_path}")

# Also export as GLB for interoperability
verts = np.asarray(mesh.vertices)
faces = np.asarray(mesh.triangles)
tmesh = trimesh.Trimesh(vertices=verts, faces=faces)
glb_path = os.path.join(args.output_dir, "tool_mesh.glb")
tmesh.export(glb_path)
print(f"  → {glb_path}")

# Volume report
if tmesh.is_watertight:
    print(f"  Tool volume: {tmesh.volume * 1e9:.2f} mm³  ({tmesh.volume * 1e6:.4f} cm³)")
else:
    print("  [WARN] Mesh is not watertight — volume estimate unreliable.")

# ──────────────────────────────────────────────────────────────────────
# Optional visualisation
# ──────────────────────────────────────────────────────────────────────
if args.visualize:
    print("[VIS] Launching Open3D viewer …")
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    mesh_vis = o3d.geometry.TriangleMesh(mesh)
    mesh_vis.paint_uniform_color([0.6, 0.8, 1.0])
    mesh_vis.compute_vertex_normals()
    o3d.visualization.draw_geometries(
        [mesh_vis, pcd, coord],
        window_name="Tool Mesh Preview",
        width=1280, height=720,
    )

print("\n[DONE] Tool capture complete.")
print(f"  Use with process_data.py:  --tool_mesh {os.path.abspath(mesh_path)}")
