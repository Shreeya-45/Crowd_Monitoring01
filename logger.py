# logger.py — CSV event logging and snapshot saving
#
# Changes from original
# ─────────────────────
# • CSV gains two new columns:
#     hull_area_m2       — real crowd footprint area (was missing before)
#     calibrated         — 1 if homography was active, 0 if fallback constant
#   This lets you audit post-hoc whether a given session had accurate geometry.

import csv
import os
import datetime
import cv2
from config import LOG_CSV, SNAPSHOT_DIR
import calibration


def _ensure_dirs():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def init_log():
    """Create CSV with header if it doesn't exist."""
    _ensure_dirs()
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "frame_idx",
                "stable_count", "density_per_m2",
                "hull_area_m2",
                "risk", "overlap_ratio", "fps",
                "calibrated"
            ])


def log_frame(frame_idx, stable_count, density, hull_area_m2,
              risk, overlap_ratio, fps):
    """Append one row per frame to the CSV."""
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    cal = 1 if calibration.is_calibrated() else 0
    with open(LOG_CSV, "a", newline="") as f:
        csv.writer(f).writerow([
            ts, frame_idx, stable_count,
            f"{density:.3f}",
            f"{hull_area_m2:.3f}",
            risk,
            f"{overlap_ratio:.2f}",
            f"{fps:.1f}",
            cal
        ])


def save_snapshot(frame, risk, frame_idx):
    """Save a JPEG snapshot when called by alert.py."""
    _ensure_dirs()
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fn  = os.path.join(SNAPSHOT_DIR, f"{ts}_{risk}_f{frame_idx}.jpg")
    cv2.imwrite(fn, frame)
    return fn
