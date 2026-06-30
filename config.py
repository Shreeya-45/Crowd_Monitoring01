import os
import torch
# config.py — all constants and calibration in one place

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX    = r"C:\Users\Dell\Downloads\134439-759734838_medium.mp4"
CAPTURE_W       = 1280
CAPTURE_H       = 720
DISPLAY_W       = 1280
DISPLAY_H       = 720

# ── Grid ──────────────────────────────────────────────────────────────────────
GRID_ROWS       = 4
GRID_COLS       = 4

# ── Homography calibration ────────────────────────────────────────────────────
#
# HOMOGRAPHY_FILE  — where the calibrated H matrix is saved/loaded (numpy .npy)
# WORLD_GRID_W/H   — real-world floor extent (metres) covered by the 4×4 grid.
#                    Set these to the physical width and depth of the monitored
#                    area BEFORE you run calibration.py.
#                    Example: a 10 m wide × 8 m deep room → 10.0, 8.0
#
HOMOGRAPHY_FILE  = "homography.npy"
WORLD_GRID_W     = 10.0     # metres — total floor width  the grid covers
WORLD_GRID_H     = 8.0      # metres — total floor depth  the grid covers

# ── Fallback area constant (used only when NOT calibrated) ────────────────────
#
# If no homography.npy exists the system falls back to this uniform constant,
# which is the original behaviour.  Calibrate as soon as possible for accuracy.
#
# How to measure it:  stand in the camera view, identify one grid cell on the
# floor, measure its real width and depth with a tape, multiply.
# E.g. a cell that covers 1.6 m x 1.25 m  ->  CELL_AREA_M2 = 2.0
CELL_AREA_M2    = 2.0

# ── Detection ─────────────────────────────────────────────────────────────────
# Use custom weights if training is complete, otherwise fallback to base model
MODEL_PATH = os.path.abspath(os.path.join("runs", "detect", "crowd_monitor_finetuned", "weights", "best.pt"))
if not os.path.isfile(MODEL_PATH):
    # Since we are multi-threaded now, we can safely use the smarter medium model for maximum accuracy!
    MODEL_PATH = "yolov10n.pt" if torch.cuda.is_available() else "yolov8m.pt"
    
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# ── Density CNN (Crowd Counting) ─────────────────────────────────────────────
USE_CNN_DENSITY = False
CNN_MODEL_PATH  = os.path.abspath(os.path.join("models", "crowd_cnn.pt"))
CNN_INPUT_SIZE  = (640, 640)

CONF_THRESHOLD  = 0.01  # Dropped from 0.05 to grab every possible person
ALPHA_SHAPE_PARAM = 0.7   # Alpha value for concave hull (alphashape library)
BUFFER_SIZE     = 15          # temporal smoothing frames

# ── DBSCAN ───────────────────────────────────────────────────────────────────
DBSCAN_EPS      = 1.5         # metres
DBSCAN_MIN_SAMPLES = 3

# ── Kernel Density Estimation (KDE) ──────────────────────────────────────────
KDE_RESOLUTION  = 640         # resolution of the KDE heatmap grid
KDE_BANDWIDTH   = 0.6         # influence radius of each person in metres

# ── Manual ROI (Region of Interest) ──────────────────────────────────────────
# Only people inside this polygon will be counted and monitored.
# Defined as list of (x, y) world coordinates in metres.
# If empty, the entire field of view is monitored.
MANUAL_ROI = [
    # Example: [(0.0, 0.0), (10.0, 0.0), (10.0, 8.0), (0.0, 8.0)]
]

# ── Static Obstacles ──────────────────────────────────────────────────────────
# Define polygons (list of (x, y) world coordinates in metres) for areas 
# containing furniture, machinery, or tables that should be excluded.
STATIC_OBSTACLES = [
    # Add real obstacle polygons here as (x, y) world-coordinate tuples.
    # Leave empty if there are no fixed obstacles in the monitored area.
    # Example: [(1.0, 2.0), (3.0, 2.0), (3.0, 2.5), (1.0, 2.5)]
]

# ── Occlusion correction (Paper 4 / ECD-DSA) ─────────────────────────────────
OCCLUSION_CORRECTION = True
OCCLUSION_GAIN       = 0.55  # Dense crowds: YOLO misses ~40-60% of occluded people

# ── Polus Level-of-Service density thresholds (people / m²) ──────────────────
# Source: Polus, Schofer & Ushpiz (1983), adopted in ECD-DSA Table I
DENSITY_THRESHOLDS = [
    (0.60,  "VERY LOW",  (80,  220, 100)),
    (1.00,  "LOW",       (0,   220, 255)),
    (1.80,  "MODERATE",  (0,   200, 255)),
    (2.50,  "HIGH",      (0,   140, 255)),
    (999,   "CRITICAL",  (60,  60,  255)),
]

# ── Alert ─────────────────────────────────────────────────────────────────────
ALERT_RISK_LEVELS       = {"HIGH", "CRITICAL"}
ALERT_SUSTAIN_FRAMES    = 10      # frames before alarm fires
RECORD_ON_ALERT         = True
LOG_CSV                 = "crowd_log.csv"
SNAPSHOT_DIR            = "snapshots"

# ── Flow ──────────────────────────────────────────────────────────────────────
FLOW_HISTORY_LEN        = 2       # frames kept for motion vectors
