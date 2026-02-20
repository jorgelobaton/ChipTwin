"""
Manual Segmentation Tool with Visual Feedback + Correction
==========================================================
Interactive OpenCV-based tool to manually segment any object on a video frame,
propagate through the video with SAM2, review the result, and **correct** frames
where the propagation drifted — then re-propagate.

Workflow:
  Phase 1 — ANNOTATE: mark the target object on the initial frame.
  Phase 2 — PROPAGATE: SAM2 propagates mask through all frames.
  Phase 3 — REVIEW: scrub through frames, visually inspect the mask.
      If a frame is bad, press 'c' to CORRECT it (opens annotator on that frame),
      then re-propagates from that frame onward.
      Repeat review/correct until satisfied, then press 's' to SAVE.

Keyboard reference (shown on-screen):
  Phase 1 (Annotate):
    'p'       - Point mode (left-click=positive, right-click=negative)
    'b'       - Box mode (click-drag)
    'z'       - Undo last annotation
    'r'       - Reset all annotations on this frame
    Enter     - Confirm mask -> propagate
    'q'/Esc   - Quit without saving

  Phase 3 (Review):
    left/right arrow - Previous / next frame
    Home/End  - Jump to first / last frame
    'c'       - Correct current frame (opens annotator, then re-propagates)
    Space     - Play/pause auto-playback
    's'       - Accept & save all masks
    'q'/Esc   - Quit without saving

The output format is identical to segment_util_video.py:
  mask/{camera_id}/{obj_id}/{frame_idx}.png
  mask/mask_info_{camera_id}.json
"""

import os
import sys
import cv2
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import supervision as sv
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from argparse import ArgumentParser

# ─── Args ───────────────────────────────────────────────────────────────────────
parser = ArgumentParser(description="Manually segment an object and propagate with SAM2 (with correction)")
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--case_name", type=str, required=True)
parser.add_argument("--camera_id", type=str, required=True)
parser.add_argument("--label", type=str, default="controller",
                    help="Label name for this mask (stored in mask_info json)")
parser.add_argument("--obj_id", type=int, default=None,
                    help="Object ID to assign. If None, auto-picks the next available ID.")
parser.add_argument("--output_path", type=str, default="NONE")
parser.add_argument("--frame_idx", type=int, default=0,
                    help="Frame index to annotate on (default: first frame)")
args = parser.parse_args()

base_path = args.base_path
case_name = args.case_name
camera_id = args.camera_id
if args.output_path == "NONE":
    output_path = f"{base_path}/{case_name}"
else:
    output_path = args.output_path

# Get frame_num from metadata if available
frame_num = -1
metadata_path = f"{base_path}/{case_name}/metadata.json"
if os.path.exists(metadata_path):
    with open(metadata_path, "r") as f:
        meta = json.load(f)
        frame_num = meta.get("frame_num", -1)


def exist_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# ─── SAM2 Model Setup ──────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAM2_CHECKPOINT = "./data_process/groundedSAM_checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

print("[Manual Segment] Loading SAM2 models...")
video_predictor = build_sam2_video_predictor(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT)
sam2_image_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT)
image_predictor = SAM2ImagePredictor(sam2_image_model)
print("[Manual Segment] Models loaded.")

# ─── Extract Video Frames ──────────────────────────────────────────────────────
VIDEO_PATH = f"{base_path}/{case_name}/color/{camera_id}.mp4"
exist_dir(f"{base_path}/{case_name}/tmp_data")
exist_dir(f"{base_path}/{case_name}/tmp_data/{case_name}")
exist_dir(f"{base_path}/{case_name}/tmp_data/{case_name}/{camera_id}")
SOURCE_VIDEO_FRAME_DIR = f"{base_path}/{case_name}/tmp_data/{case_name}/{camera_id}"

print(f"[Manual Segment] Extracting frames from {VIDEO_PATH}...")
video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)

source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
source_frames.mkdir(parents=True, exist_ok=True)

with sv.ImageSink(
    target_dir_path=source_frames, overwrite=True, image_name_pattern="{:05d}.jpg"
) as sink:
    for i, frame in enumerate(tqdm(frame_generator, desc="Saving Video Frames")):
        if frame_num > 0 and i >= frame_num:
            break
        sink.save_image(frame)

frame_names = sorted(
    [p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
     if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]],
    key=lambda p: int(os.path.splitext(p)[0])
)
num_total_frames = len(frame_names)
print(f"[Manual Segment] {num_total_frames} frames extracted.")


def load_frame(frame_idx):
    """Load a video frame by index."""
    path = os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx])
    return cv2.imread(path)


# ─── Visual constants ──────────────────────────────────────────────────────────
COLORS = {
    "mask_overlay": (0, 255, 0),         # green
    "correction_overlay": (0, 200, 255), # orange for corrected-frame markers
    "positive_point": (0, 255, 0),       # green
    "negative_point": (0, 0, 255),       # red
    "bbox": (255, 200, 0),              # cyan-ish
    "text": (255, 255, 255),
    "text_bg": (40, 40, 40),
    "timeline_bg": (30, 30, 30),
    "timeline_pos": (0, 180, 255),       # orange cursor
    "timeline_corrected": (0, 200, 255), # corrected frame dots
    "timeline_initial": (0, 255, 0),     # initial annotated frame dot
}
ALPHA = 0.45
POINT_RADIUS = 6
WINDOW_NAME = "Manual Segmentation - ChipTwin"


# ─── ManualAnnotator (per-frame interactive annotation) ─────────────────────────

class ManualAnnotator:
    """Interactive OpenCV annotation tool with live SAM2 mask preview.

    Can operate on any frame.  Supports undo.
    """

    def __init__(self, image: np.ndarray, image_predictor: SAM2ImagePredictor,
                 frame_idx: int = 0, label: str = "", existing_mask: np.ndarray = None):
        self.original = image.copy()
        self.h, self.w = image.shape[:2]
        self.predictor = image_predictor
        self.predictor.set_image(image)
        self.frame_idx = frame_idx
        self.label = label

        # Annotation state
        self.mode = "point"  # "point" or "box"
        self.positive_points = []
        self.negative_points = []
        self.bbox = None
        self.bbox_start = None
        self.dragging = False

        # Undo history: list of (action_type, data) tuples
        self._history = []

        # Current predicted mask
        self.mask = existing_mask.copy() if existing_mask is not None else None
        self.confirmed = False
        self.cancelled = False

    # ── SAM2 prediction ─────────────────────────────────────────────────────────
    def _predict_mask(self):
        has_points = len(self.positive_points) > 0 or len(self.negative_points) > 0
        has_box = self.bbox is not None

        if not has_points and not has_box:
            self.mask = None
            return

        point_coords = None
        point_labels = None
        if has_points:
            all_pts = self.positive_points + self.negative_points
            all_labels = [1] * len(self.positive_points) + [0] * len(self.negative_points)
            point_coords = np.array(all_pts, dtype=np.float32)
            point_labels = np.array(all_labels, dtype=np.int32)

        box_input = None
        if has_box:
            x1, y1, x2, y2 = self.bbox
            box_input = np.array([[x1, y1, x2, y2]], dtype=np.float32)

        with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_input,
                multimask_output=True,
            )

        best_idx = np.argmax(scores)
        self.mask = masks[best_idx].astype(bool)

    # ── Undo ────────────────────────────────────────────────────────────────────
    def _undo(self):
        if not self._history:
            print("[Undo] Nothing to undo")
            return
        action_type, _ = self._history.pop()
        if action_type == "positive_point" and self.positive_points:
            self.positive_points.pop()
        elif action_type == "negative_point" and self.negative_points:
            self.negative_points.pop()
        elif action_type == "bbox":
            self.bbox = None
        self._predict_mask()
        print(f"[Undo] Removed last {action_type}")

    # ── Render ──────────────────────────────────────────────────────────────────
    def _render(self, mouse_pos=None):
        vis = self.original.copy()

        # Mask overlay
        if self.mask is not None:
            overlay = vis.copy()
            overlay[self.mask] = COLORS["mask_overlay"]
            vis = cv2.addWeighted(overlay, ALPHA, vis, 1 - ALPHA, 0)
            mask_uint8 = self.mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, COLORS["mask_overlay"], 2)

        # Points
        for pt in self.positive_points:
            cv2.circle(vis, pt, POINT_RADIUS, COLORS["positive_point"], -1)
            cv2.circle(vis, pt, POINT_RADIUS + 2, (255, 255, 255), 1)
        for pt in self.negative_points:
            cv2.circle(vis, pt, POINT_RADIUS, COLORS["negative_point"], -1)
            cv2.circle(vis, pt, POINT_RADIUS + 2, (255, 255, 255), 1)
            cv2.line(vis, (pt[0]-4, pt[1]-4), (pt[0]+4, pt[1]+4), (255, 255, 255), 1)
            cv2.line(vis, (pt[0]-4, pt[1]+4), (pt[0]+4, pt[1]-4), (255, 255, 255), 1)

        # Bounding box
        if self.bbox is not None:
            cv2.rectangle(vis, (self.bbox[0], self.bbox[1]),
                          (self.bbox[2], self.bbox[3]), COLORS["bbox"], 2)
        elif self.dragging and self.bbox_start is not None and mouse_pos is not None:
            cv2.rectangle(vis, self.bbox_start, mouse_pos, COLORS["bbox"], 2)

        # HUD
        hud_lines = [
            f"ANNOTATE  |  Frame {self.frame_idx}/{num_total_frames-1}  |  '{self.label}'",
            f"Mode: {self.mode.upper()} ('p'=point, 'b'=box)",
            "Left-click=positive | Right-click=negative" if self.mode == "point"
            else "Click-drag: draw box",
            f"Points: {len(self.positive_points)}+ / {len(self.negative_points)}-",
            "'z'=undo | 'r'=reset | Enter=confirm | 'q'/Esc=cancel",
        ]
        if self.mask is not None:
            area_pct = self.mask.sum() / (self.h * self.w) * 100
            hud_lines.append(f"Mask area: {area_pct:.1f}%")

        y_offset = 25
        for line in hud_lines:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(vis, (8, y_offset - th - 4), (tw + 16, y_offset + 6),
                          COLORS["text_bg"], -1)
            cv2.putText(vis, line, (12, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        COLORS["text"], 1, cv2.LINE_AA)
            y_offset += 26

        return vis

    # ── Mouse callback ──────────────────────────────────────────────────────────
    def _on_mouse(self, event, x, y, flags, param):
        if self.mode == "point":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.positive_points.append((x, y))
                self._history.append(("positive_point", (x, y)))
                self._predict_mask()
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.negative_points.append((x, y))
                self._history.append(("negative_point", (x, y)))
                self._predict_mask()
        elif self.mode == "box":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.bbox_start = (x, y)
                self.dragging = True
            elif event == cv2.EVENT_LBUTTONUP and self.dragging:
                self.dragging = False
                if self.bbox_start is not None:
                    x1, y1 = self.bbox_start
                    x2, y2 = x, y
                    self.bbox = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
                    self._history.append(("bbox", self.bbox))
                    self.bbox_start = None
                    self._predict_mask()

    # ── Main loop ───────────────────────────────────────────────────────────────
    def run(self):
        """Main interaction loop. Returns the confirmed mask or None."""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        scale = min(1.0, 1280 / self.w, 960 / self.h)
        cv2.resizeWindow(WINDOW_NAME, int(self.w * scale), int(self.h * scale))

        mouse_pos = [None]

        def _mouse_cb(event, x, y, flags, param):
            self._on_mouse(event, x, y, flags, param)
            mouse_pos[0] = (x, y)

        cv2.setMouseCallback(WINDOW_NAME, _mouse_cb)

        print("\n" + "=" * 60)
        print(f" ANNOTATE  |  Frame {self.frame_idx}  |  '{self.label}'")
        print("=" * 60)
        print(" 'p' = Point mode | 'b' = Box mode")
        print(" Left-click = positive | Right-click = negative")
        print(" 'z' = undo | 'r' = reset | Enter = confirm | q/Esc = cancel")
        print("=" * 60 + "\n")

        while True:
            vis = self._render(mouse_pos=mouse_pos[0])
            cv2.imshow(WINDOW_NAME, vis)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('p'):
                self.mode = "point"
                print("[Mode] POINT")
            elif key == ord('b'):
                self.mode = "box"
                print("[Mode] BOX")
            elif key == ord('z'):
                self._undo()
            elif key == ord('r'):
                self.positive_points.clear()
                self.negative_points.clear()
                self.bbox = None
                self.bbox_start = None
                self.dragging = False
                self.mask = None
                self._history.clear()
                print("[Reset] All annotations cleared")
            elif key == 13:  # Enter
                if self.mask is not None:
                    self.confirmed = True
                    break
                else:
                    print("[!] No mask to confirm. Add annotations first.")
            elif key == ord('q') or key == 27:
                self.cancelled = True
                break

        # Don't destroy the window here — reused by reviewer
        if self.confirmed:
            return self.mask
        return None


# ─── Review & Correct ──────────────────────────────────────────────────────────

class MaskReviewer:
    """Scrub through propagated masks with timeline, and correct bad frames."""

    def __init__(self, video_segments: dict, obj_id: int, label: str,
                 corrected_frames: set, initial_frame: int):
        self.video_segments = video_segments
        self.obj_id = obj_id
        self.label = label
        self.corrected_frames = corrected_frames
        self.initial_frame = initial_frame
        self.frame_indices = sorted(video_segments.keys())
        self.current_pos = 0
        self.playing = False
        self.action = None
        self.correct_frame_idx = None

    def _get_mask_for_frame(self, frame_idx):
        if frame_idx in self.video_segments:
            seg = self.video_segments[frame_idx]
            if self.obj_id in seg:
                return seg[self.obj_id][0]  # (1, H, W) -> (H, W)
        return None

    def _render_frame(self, frame_idx):
        frame = load_frame(frame_idx)
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)

        h, w = frame.shape[:2]
        mask = self._get_mask_for_frame(frame_idx)

        if mask is not None:
            is_corrected = frame_idx in self.corrected_frames
            color = COLORS["correction_overlay"] if is_corrected else COLORS["mask_overlay"]

            overlay = frame.copy()
            overlay[mask] = color
            frame = cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0)

            mask_uint8 = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, color, 2)

            area_pct = mask.sum() / (h * w) * 100
        else:
            area_pct = 0.0

        # HUD
        corr_tag = "  [CORRECTED]" if frame_idx in self.corrected_frames else ""
        init_tag = "  [INITIAL]" if frame_idx == self.initial_frame else ""
        hud_lines = [
            f"REVIEW  |  Frame {frame_idx}/{num_total_frames-1}  |  '{self.label}' (id={self.obj_id}){corr_tag}{init_tag}",
            f"Mask: {area_pct:.1f}%  |  {'>> PLAYING' if self.playing else '|| PAUSED'}  |  Corrections: {len(self.corrected_frames)}",
            "Left/Right=nav | Space=play | Home/End=jump | 'c'=correct | 's'=save | q=quit",
        ]

        y_offset = 25
        for line in hud_lines:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (8, y_offset - th - 4), (tw + 16, y_offset + 6),
                          COLORS["text_bg"], -1)
            cv2.putText(frame, line, (12, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        COLORS["text"], 1, cv2.LINE_AA)
            y_offset += 26

        # ── Timeline bar at the bottom ──
        bar_h = 30
        bar_y = h - bar_h
        cv2.rectangle(frame, (0, bar_y), (w, h), COLORS["timeline_bg"], -1)

        n = len(self.frame_indices)
        if n > 1:
            margin = 10
            usable_w = w - 2 * margin

            # Markers for special frames
            for fidx in self.corrected_frames:
                if fidx in self.frame_indices:
                    idx = self.frame_indices.index(fidx)
                    x = int(idx / (n - 1) * usable_w) + margin
                    cv2.circle(frame, (x, bar_y + bar_h // 2), 5,
                               COLORS["timeline_corrected"], -1)

            # Initial frame marker
            if self.initial_frame in self.frame_indices:
                idx = self.frame_indices.index(self.initial_frame)
                x = int(idx / (n - 1) * usable_w) + margin
                cv2.circle(frame, (x, bar_y + bar_h // 2), 5,
                           COLORS["timeline_initial"], -1)

            # Thin progress line
            cx = int(self.current_pos / (n - 1) * usable_w) + margin
            cv2.line(frame, (margin, bar_y + bar_h // 2), (cx, bar_y + bar_h // 2),
                     COLORS["timeline_pos"], 2)

            # Current position cursor
            cv2.rectangle(frame, (cx - 3, bar_y + 3), (cx + 3, bar_y + bar_h - 3),
                          COLORS["timeline_pos"], -1)

            # Frame number next to cursor
            cv2.putText(frame, str(frame_idx), (cx + 8, bar_y + bar_h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["text"], 1)

        return frame

    def run(self):
        """Main review loop. Returns action: 'save', 'correct', or 'quit'."""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        sample = load_frame(0)
        h, w = sample.shape[:2]
        scale = min(1.0, 1280 / w, 960 / h)
        cv2.resizeWindow(WINDOW_NAME, int(w * scale), int(h * scale))

        print("\n" + "=" * 60)
        print(" REVIEW  |  Scrub through frames to check mask quality")
        print("=" * 60)
        print(" Left/Right = navigate | Space = play/pause")
        print(" Home/End = first/last frame")
        print(" 'c' = correct this frame | 's' = save | q/Esc = quit")
        print("=" * 60 + "\n")

        while True:
            frame_idx = self.frame_indices[self.current_pos]
            vis = self._render_frame(frame_idx)
            cv2.imshow(WINDOW_NAME, vis)

            wait_ms = 50 if self.playing else 30
            key = cv2.waitKey(wait_ms) & 0xFF

            if self.playing and key == 255:
                # Auto-advance
                if self.current_pos < len(self.frame_indices) - 1:
                    self.current_pos += 1
                else:
                    self.playing = False
                continue

            # Arrow keys: OpenCV encodes them differently across platforms.
            # On Linux with GTK backend: Left=81, Right=83, Up=82, Down=84
            # With Qt: Left=0x250000, etc. We handle the common cases.
            if key == 81 or key == 2:  # Left arrow
                self.playing = False
                self.current_pos = max(0, self.current_pos - 1)
            elif key == 83 or key == 3:  # Right arrow
                self.playing = False
                self.current_pos = min(len(self.frame_indices) - 1, self.current_pos + 1)
            elif key == 80 or key == 0:  # Home
                self.current_pos = 0
                self.playing = False
            elif key == 87 or key == 1:  # End
                self.current_pos = len(self.frame_indices) - 1
                self.playing = False
            elif key == ord(' '):
                self.playing = not self.playing
            elif key == ord('c'):
                self.playing = False
                self.action = "correct"
                self.correct_frame_idx = frame_idx
                print(f"[Review] Will correct frame {frame_idx}")
                return "correct"
            elif key == ord('s'):
                self.action = "save"
                return "save"
            elif key == ord('q') or key == 27:
                self.action = "quit"
                return "quit"


# ─── Propagation helpers ───────────────────────────────────────────────────────

def propagate_video(video_pred, inference_state, obj_id):
    """Run full SAM2 video propagation. Returns {frame_idx: {obj_id: mask_array}}."""
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_pred.propagate_in_video(
            inference_state):
        video_segments[out_frame_idx] = {
            oid: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, oid in enumerate(out_obj_ids)
        }
    return video_segments


def register_mask_as_prompts(video_pred, inference_state, frame_idx, obj_id,
                              mask, positive_points=None, negative_points=None):
    """Register a confirmed mask with SAM2 video predictor using box + point prompts."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return
    bbox = np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

    video_pred.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=frame_idx,
        obj_id=obj_id,
        box=bbox,
    )

    # Add user-clicked points for more robust propagation
    if positive_points or negative_points:
        pts_list = list(positive_points or [])
        labels_list = [1] * len(pts_list)
        if negative_points:
            pts_list.extend(negative_points)
            labels_list.extend([0] * len(negative_points))
        video_pred.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=np.array(pts_list, dtype=np.float32),
            labels=np.array(labels_list, dtype=np.int32),
        )


# ─── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ann_frame_idx = args.frame_idx

    image = load_frame(ann_frame_idx)
    if image is None:
        print(f"[ERROR] Could not load frame {ann_frame_idx}")
        sys.exit(1)

    print(f"[Manual Segment] Annotating frame {ann_frame_idx}: '{args.label}'")

    # ── Phase 1: Initial annotation ──
    annotator = ManualAnnotator(image, image_predictor,
                                frame_idx=ann_frame_idx, label=args.label)
    mask = annotator.run()

    if mask is None:
        print("[Cancelled] No mask confirmed. Exiting.")
        cv2.destroyAllWindows()
        sys.exit(0)

    print(f"[Confirmed] Mask on frame {ann_frame_idx}: {mask.sum()} pixels")

    # ── Determine object ID ──
    mask_dir = f"{output_path}/mask"
    mask_info_path = f"{mask_dir}/mask_info_{camera_id}.json"

    existing_info = {}
    if os.path.exists(mask_info_path):
        with open(mask_info_path, "r") as f:
            existing_info = json.load(f)

    if args.obj_id is not None:
        obj_id = args.obj_id
    else:
        used_ids = [int(k) for k in existing_info.keys()] if existing_info else []
        obj_id = max(used_ids) + 1 if used_ids else 0

    print(f"[Manual Segment] Object ID = {obj_id}, label = '{args.label}'")

    # ── Enable autocast ──
    torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ── Phase 2: Initial propagation ──
    print("[Manual Segment] Initializing SAM2 video predictor...")
    inference_state = video_predictor.init_state(video_path=SOURCE_VIDEO_FRAME_DIR)

    register_mask_as_prompts(
        video_predictor, inference_state, ann_frame_idx, obj_id, mask,
        positive_points=annotator.positive_points,
        negative_points=annotator.negative_points,
    )

    print("[Manual Segment] Propagating mask through video...")
    video_segments = propagate_video(video_predictor, inference_state, obj_id)
    print(f"[Manual Segment] Propagated to {len(video_segments)} frames.")

    # ── Correction state ──
    # Stores all correction masks so we can re-register them after reset_state
    correction_masks = {}      # {frame_idx: (mask, pos_points, neg_points)}
    corrected_frames = set()   # for visual feedback in reviewer

    # ── Phase 3: Review & Correct loop ──
    while True:
        reviewer = MaskReviewer(video_segments, obj_id, args.label,
                                corrected_frames, initial_frame=ann_frame_idx)
        action = reviewer.run()

        if action == "save":
            print("[Review] Saving masks...")
            break

        elif action == "correct":
            corr_frame_idx = reviewer.correct_frame_idx
            print(f"\n[Correct] Opening annotator on frame {corr_frame_idx}...")

            corr_image = load_frame(corr_frame_idx)
            existing_mask = reviewer._get_mask_for_frame(corr_frame_idx)

            corr_annotator = ManualAnnotator(
                corr_image, image_predictor,
                frame_idx=corr_frame_idx, label=args.label,
                existing_mask=existing_mask,
            )
            corr_mask = corr_annotator.run()

            if corr_mask is None:
                print("[Correct] Cancelled, returning to review.")
                continue

            print(f"[Correct] Confirmed correction on frame {corr_frame_idx}: "
                  f"{corr_mask.sum()} pixels")
            corrected_frames.add(corr_frame_idx)
            correction_masks[corr_frame_idx] = (
                corr_mask,
                list(corr_annotator.positive_points),
                list(corr_annotator.negative_points),
            )

            # Reset SAM2 and re-register ALL conditioning frames, then re-propagate
            print("[Correct] Re-initializing video predictor with all annotations...")
            video_predictor.reset_state(inference_state)

            # Re-register initial annotation
            register_mask_as_prompts(
                video_predictor, inference_state, ann_frame_idx, obj_id, mask,
                positive_points=annotator.positive_points,
                negative_points=annotator.negative_points,
            )

            # Re-register all accumulated corrections
            for cf_idx, (cf_mask, cf_pos, cf_neg) in correction_masks.items():
                register_mask_as_prompts(
                    video_predictor, inference_state, cf_idx, obj_id, cf_mask,
                    positive_points=cf_pos,
                    negative_points=cf_neg,
                )

            print("[Correct] Re-propagating...")
            video_segments = propagate_video(video_predictor, inference_state, obj_id)
            print(f"[Correct] Re-propagated to {len(video_segments)} frames.")

        elif action == "quit":
            print("[Quit] Exiting without saving.")
            cv2.destroyAllWindows()
            os.system(f"rm -rf {base_path}/{case_name}/tmp_data")
            sys.exit(0)

    # ── Save masks (identical format to segment_util_video.py) ──
    cv2.destroyAllWindows()

    exist_dir(mask_dir)
    exist_dir(f"{mask_dir}/{camera_id}")
    exist_dir(f"{mask_dir}/{camera_id}/{obj_id}")

    for frame_idx_out, masks_dict in video_segments.items():
        for oid, m in masks_dict.items():
            save_dir = f"{mask_dir}/{camera_id}/{oid}"
            exist_dir(save_dir)
            Image.fromarray((m[0] * 255).astype(np.uint8)).save(
                f"{save_dir}/{frame_idx_out}.png"
            )

    # Update mask_info json
    existing_info[str(obj_id)] = args.label
    with open(mask_info_path, "w") as f:
        json.dump(existing_info, f)

    print(f"[Saved] Masks  -> {mask_dir}/{camera_id}/{obj_id}/")
    print(f"[Saved] Info   -> {mask_info_path}")

    # Cleanup
    print("[Manual Segment] Cleaning up temporary frames...")
    os.system(f"rm -rf {base_path}/{case_name}/tmp_data")
    print("[Manual Segment] Done!")
