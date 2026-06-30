# detector.py — YOLO tracking + sliced tile inference + occlusion-aware count

import numpy as np
import torch
from ultralytics import YOLO
import calibration
from config import (MODEL_PATH, CONF_THRESHOLD, GRID_ROWS, GRID_COLS,
                    OCCLUSION_CORRECTION, OCCLUSION_GAIN, DEVICE)
from head_localizer import HeadLocalizer

# Adapt to available hardware
_ON_GPU      = torch.cuda.is_available()
_INFER_SIZE  = 1280 if _ON_GPU else 960   # full-frame inference resolution
_TILE_SIZE   = 640  if _ON_GPU else 480   # tile size for sliced pass
_TILE_ENABLE = True                        # set False to disable tile pass


def load_model():
    return YOLO(MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Overlap ratio (for occlusion correction)
# ─────────────────────────────────────────────────────────────────────────────

def compute_overlap_ratio(boxes):
    """
    Fraction of detected boxes whose center falls inside another box.
    High overlap → occlusion → YOLO recall drops (ECD-DSA: ~42% in dense).
    """
    n = len(boxes)
    if n < 2:
        return 0.0
    overlap = 0
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cx, cy = (x1+x2)/2, (y1+y2)/2
        for j, (ox1, oy1, ox2, oy2) in enumerate(boxes):
            if i == j:
                continue
            if ox1 <= cx <= ox2 and oy1 <= cy <= oy2:
                overlap += 1
                break
    return overlap / n


# ─────────────────────────────────────────────────────────────────────────────
# Sliced tile inference  (detects small / distant people missed by full-frame)
# ─────────────────────────────────────────────────────────────────────────────

def _iou(a, b):
    """IoU between two boxes (x1,y1,x2,y2)."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def _nms(boxes, iou_thresh=0.5):
    """Simple NMS. boxes = list of (x1,y1,x2,y2,conf)."""
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    kept = []
    for box in boxes:
        if all(_iou(box, k) < iou_thresh for k in kept):
            kept.append(box)
    return kept


def _detect_tiled(model, frame, tile_size=240, overlap=0.5, iou_thresh=0.6):
    """
    Run model.predict() on overlapping tiles and return merged bounding boxes
    in full-frame pixel coordinates. By using an extremely small tile_size (e.g. 240)
    but inferring at 640, YOLO effectively zooms in nearly 3x, finding microscopic background people.

    This dramatically improves recall for:
      • Small/distant people at the back of the scene (30-50 px tall)
      • People near frame edges that full-frame NMS suppresses

    Returns: list of (x1, y1, x2, y2, conf) tuples
    """
    h, w = frame.shape[:2]
    stride = int(tile_size * (1 - overlap))
    raw = []

    # Build tile origins — always include an extra tile that covers the right/bottom edge
    xs = list(range(0, max(w - tile_size, 1), stride)) + [max(0, w - tile_size)]
    ys = list(range(0, max(h - tile_size, 1), stride)) + [max(0, h - tile_size)]
    xs = sorted(set(xs)); ys = sorted(set(ys))

    for y0 in ys:
        for x0 in xs:
            x0 = min(x0, max(0, w - tile_size))
            y0_c = min(y0, max(0, h - tile_size))
            tile = frame[y0_c:y0_c+tile_size, x0:x0+tile_size]
            if tile.size == 0:
                continue

            results = model.predict(
                tile,
                classes=[0],
                conf=CONF_THRESHOLD,
                iou=iou_thresh,
                imgsz=640,  # Fixed to 640 so smaller tiles are upscaled (zoom effect)
                device=DEVICE,
                verbose=False
            )
            if results[0].boxes is None:
                continue
            for box in results[0].boxes:
                if model.names[int(box.cls[0])] != "person":
                    continue
                bx1, by1, bx2, by2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                # Map tile coords → full-frame coords
                raw.append((bx1 + x0, by1 + y0_c,
                             bx2 + x0, by2 + y0_c, conf))

    return _nms(raw, iou_thresh=iou_thresh)


# ─────────────────────────────────────────────────────────────────────────────
# Main detection entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(model, frame, frame_h, frame_w):
    """
    Two-pass detection:
      Pass 1 — model.track() on the full frame  → gives person IDs + tracks
      Pass 2 — tiled model.predict()            → finds small/distant people
                                                   missed by the full-frame pass

    Detections from both passes are merged (NMS deduplication).

    Returns:
        detections  — list of dicts: x1 y1 x2 y2 cx cy fx fy wx wy row col pid
        raw_count   — merged detection count before occlusion correction
        corr_count  — occlusion-corrected float count
        overlap_r   — overlap ratio (0–1)
        zone_counts — 2-D list [GRID_ROWS][GRID_COLS]
    """
    cell_w = frame_w // GRID_COLS
    cell_h = frame_h // GRID_ROWS

    # ── Pass 1: full-frame tracking ───────────────────────────────────────
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=CONF_THRESHOLD,
        iou=0.6,        # Extreme IoU -> NMS keeps highly overlapping boxes (dense crowds)
        imgsz=_INFER_SIZE,  # 1280 on GPU, 960 on CPU (faster but still good)
        device=DEVICE,
        verbose=False
    )

    detections  = []
    zone_counts = [[0]*GRID_COLS for _ in range(GRID_ROWS)]
    boxes_raw   = []    # (x1,y1,x2,y2) for overlap ratio
    tracked_boxes = []  # (x1,y1,x2,y2,conf) for NMS dedup with tile pass

    if results[0].boxes is not None:
        for box in results[0].boxes:
            if model.names[int(box.cls[0])] != "person":
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cx = (x1+x2)//2
            cy = (y1+y2)//2

            fx, fy, _ = HeadLocalizer.compute_foot_point(x1, y1, x2, y2)
            fx, fy = int(fx), int(fy)

            wx, wy = None, None
            if calibration.is_calibrated():
                try:
                    world_pt = calibration.px_to_world([(fx, fy)])
                    wx, wy = float(world_pt[0, 0]), float(world_pt[0, 1])
                except Exception:
                    pass

            if calibration.is_calibrated() and wx is not None and wy is not None:
                row, col = calibration.world_to_grid(wx, wy)
            else:
                col = min(max(fx, 0) // cell_w, GRID_COLS-1)
                row = min(max(fy, 0) // cell_h, GRID_ROWS-1)

            pid = int(box.id[0]) if box.id is not None else -1
            zone_counts[row][col] += 1
            boxes_raw.append((x1, y1, x2, y2))
            tracked_boxes.append((x1, y1, x2, y2, conf))

            detections.append(dict(x1=x1, y1=y1, x2=x2, y2=y2,
                                   cx=cx, cy=cy, fx=fx, fy=fy,
                                   wx=wx, wy=wy,
                                   row=row, col=col, pid=pid))

    # ── Pass 2: tiled detection — find missed small/distant people ────────
    # Dynamically scale settings: dense crowds need EXTREME zoom and overlap tolerance.
    is_dense = len(tracked_boxes) > 10
    dyn_tile = 240 if is_dense else 640
    dyn_iou  = 0.60 if is_dense else 0.40

    tile_boxes = _detect_tiled(model, frame, tile_size=dyn_tile, iou_thresh=dyn_iou, overlap=0.5) if _TILE_ENABLE else []

    for (tx1, ty1, tx2, ty2, tconf) in tile_boxes:
        # Skip if this box overlaps with an already-tracked detection
        if any(_iou((tx1, ty1, tx2, ty2), (bx1, by1, bx2, by2)) > dyn_iou
               for (bx1, by1, bx2, by2, _) in tracked_boxes):
            continue

        x1, y1, x2, y2 = int(tx1), int(ty1), int(tx2), int(ty2)
        cx = (x1+x2)//2
        cy = (y1+y2)//2

        fx, fy, _ = HeadLocalizer.compute_foot_point(x1, y1, x2, y2)
        fx, fy = int(fx), int(fy)

        wx, wy = None, None
        if calibration.is_calibrated():
            try:
                world_pt = calibration.px_to_world([(fx, fy)])
                wx, wy = float(world_pt[0, 0]), float(world_pt[0, 1])
            except Exception:
                pass

        if calibration.is_calibrated() and wx is not None and wy is not None:
            row, col = calibration.world_to_grid(wx, wy)
        else:
            col = min(max(fx, 0) // cell_w, GRID_COLS-1)
            row = min(max(fy, 0) // cell_h, GRID_ROWS-1)

        zone_counts[row][col] += 1
        boxes_raw.append((x1, y1, x2, y2))

        detections.append(dict(x1=x1, y1=y1, x2=x2, y2=y2,
                               cx=cx, cy=cy, fx=fx, fy=fy,
                               wx=wx, wy=wy,
                               row=row, col=col, pid=-1))

    # ── Occlusion correction ──────────────────────────────────────────────
    raw_count  = len(detections)
    overlap_r  = compute_overlap_ratio(boxes_raw)
    corr_count = (raw_count * (1 + OCCLUSION_GAIN * overlap_r)
                  if OCCLUSION_CORRECTION else float(raw_count))

    return detections, raw_count, corr_count, overlap_r, zone_counts
