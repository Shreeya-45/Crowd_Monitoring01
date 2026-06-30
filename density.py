# density.py
# Context-aware crowd density calculation
#
# Changes from original
# ─────────────────────
# • _meters_per_pixel() is REMOVED — it assumed an overhead camera and
#   produced a single uniform scale factor that is wrong for angled cameras.
#
# • convex_hull_area_m2() now:
#     - uses wx, wy (world metres) from each detection, not pixel centroids
#     - computes the hull area directly in m² via cv2.contourArea on world pts
#     - returns hull_pts in BOTH world coords (for area) and pixel coords (for
#       drawing on the frame)
#
# • compute_cell_densities() uses calibration.cell_area_m2(r, c) for each
#   cell individually — cells near the camera and far from it now get their
#   own correct real-world area instead of sharing one constant.
#
# • DensityTracker.update() passes world coords through and also returns
#   the pixel hull_pts separately so the UI can draw them unmodified.

import cv2
import numpy as np
import alphashape
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon, MultiPolygon, Point
from collections import deque

from config import (
    BUFFER_SIZE,
    DENSITY_THRESHOLDS,
    DBSCAN_EPS, DBSCAN_MIN_SAMPLES,
    GRID_ROWS, # Keep this line
    GRID_COLS, # Keep this line
    ALPHA_SHAPE_PARAM, # Add this line
    KDE_RESOLUTION,
    KDE_BANDWIDTH,
    STATIC_OBSTACLES,
    MANUAL_ROI,
    WORLD_GRID_W,
    WORLD_GRID_H
)
from context_risk import get_normalized_density
import calibration

# Pre-initialize obstacle polygons for performance
OBSTACLE_POLYGONS = [Polygon(pts) for pts in STATIC_OBSTACLES] if STATIC_OBSTACLES else []
ROI_POLYGON = Polygon(MANUAL_ROI) if MANUAL_ROI else None

# Calculate total usable area once
_room_area = WORLD_GRID_W * WORLD_GRID_H
if ROI_POLYGON:
    _room_area = ROI_POLYGON.area
    for obs in OBSTACLE_POLYGONS:
        if ROI_POLYGON.intersects(obs):
            _room_area -= ROI_POLYGON.intersection(obs).area
else:
    _room_area -= sum(p.area for p in OBSTACLE_POLYGONS)
TOTAL_USABLE_AREA = max(_room_area, 0.1)

PERSONAL_SPACE_M2 = 1.5  # Real-world area (m2) assigned to an isolated individual


# ─────────────────────────────────────────────────────────────────────────────
# Crowd footprint area  —  operates in world space (metres)
# ─────────────────────────────────────────────────────────────────────────────

def calculate_crowd_area_m2(detections):
    """
    Compute the crowd footprint area in real-world m² using Alpha Shapes (Concave Hull).

    ✓ Uses Alpha Shape (Concave Hull) to create a tight boundary around the detected crowd.

    Uses wx, wy (metres) from each detection when the system is calibrated,
    so the area is perspective-correct regardless of camera angle.

    When not calibrated, uses pixel-space hull scaled by monitored area extent
    (WORLD_GRID_W × WORLD_GRID_H) as the real-world reference.

    Returns:
        hull_area_m2  — float, real floor area in m²
        hull_pts_px   — list of np.ndarray (N,2) int32 pixel coords for drawing
        hull_pts_w    — np.ndarray (N,2) float64 world coords, or None
        hull_type     — str, "DBSCAN + Alpha Shape"
        alpha_value   — float, the alpha parameter used for alpha shape
    """
    # 0. Filter by Manual ROI if defined
    if ROI_POLYGON and calibration.is_calibrated():
        detections = [d for d in detections if ROI_POLYGON.contains(Point(d["wx"], d["wy"]))]

    n = len(detections)

    if n == 0:
        return 0.0, [], None, "None", ALPHA_SHAPE_PARAM

    if n < 2:
        min_area = n * 0.5
        return float(min_area), [], None, "None", ALPHA_SHAPE_PARAM

    calibrated = calibration.is_calibrated()
    alpha = ALPHA_SHAPE_PARAM # Use configurable alpha parameter

    def get_shape_data(pts):
        """Tries Alpha Shape, returns (area, ordered_points, shape_obj)."""
        try:
            shape = alphashape.alphashape(pts, alpha)
            if shape.is_empty:
                return None, None, None
            
            area = shape.area
            if isinstance(shape, Polygon):
                res_pts = np.array(shape.exterior.coords, dtype=np.float32)
            else: # MultiPolygon case
                # For drawing, we select the largest cluster to provide a continuous line
                largest_poly = max(shape.geoms, key=lambda p: p.area)
                res_pts = np.array(largest_poly.exterior.coords, dtype=np.float32)
            return float(area), res_pts, shape
        except Exception:
            return None, None, None

    # 1. Coordinate Extraction
    if calibrated:
        pts = np.array(
            [[d["wx"], d["wy"]] for d in detections],
            dtype=np.float32
        )
        m2_per_px2 = 1.0
        eps = DBSCAN_EPS
    else:
        # Use foot point (ground contact) — NOT bounding-box center —
        # so the hull and clusters map onto the floor, not buildings/sky.
        pts = np.array(
            [[d["fx"], d["fy"]] for d in detections],
            dtype=np.float32
        )
        from config import CAPTURE_W, CAPTURE_H, WORLD_GRID_W, WORLD_GRID_H
        eps = DBSCAN_EPS * (CAPTURE_W / max(WORLD_GRID_W, 1))  # Approx pixel eps
        total_area_m2 = WORLD_GRID_W * WORLD_GRID_H
        total_area_px2 = CAPTURE_W * CAPTURE_H
        m2_per_px2 = total_area_m2 / max(total_area_px2, 1)

    # 2. DBSCAN Clustering
    clustering = DBSCAN(eps=eps, min_samples=DBSCAN_MIN_SAMPLES).fit(pts)
    labels = clustering.labels_

    hull_area_m2 = 0.0
    hull_pts_px = []

    # 3. Process clusters vs isolated individuals
    for label in set(labels):
        cluster_mask = (labels == label)
        cluster_pts = pts[cluster_mask]
        num_in_cluster = len(cluster_pts)

        if label == -1 or num_in_cluster < 3:
            # ISOLATED PERSONS: Add fixed personal space area
            hull_area_m2 += num_in_cluster * PERSONAL_SPACE_M2
            continue

        # CROWD CLUSTERS: Use Alpha Shape (Concave Hull)
        area, res_pts, shape_obj = get_shape_data(cluster_pts)
        if area is not None:
            cluster_area = area * m2_per_px2

            if calibrated:
                for obs_poly in OBSTACLE_POLYGONS:
                    if shape_obj.intersects(obs_poly):
                        cluster_area -= shape_obj.intersection(obs_poly).area

            # Logical minimum: cluster shouldn't be smaller than person count * tight standing area
            hull_area_m2 += max(num_in_cluster * 0.35, cluster_area)

            if calibrated:
                px_pts = calibration.world_to_px(res_pts).astype(np.int32)
            else:
                px_pts = res_pts.astype(np.int32)
            hull_pts_px.append(px_pts)
        else:
            # Fallback for cluster if alphashape fails
            hull_area_m2 += num_in_cluster * PERSONAL_SPACE_M2

    # Final safety floor
    hull_area_m2 = max(hull_area_m2, n * 0.35)
    return hull_area_m2, hull_pts_px, pts, "DBSCAN + Alpha Shape", alpha

def compute_kde_heatmap(detections):
    """
    Generates a continuous Kernel Density Estimation map.
    Returns a normalized 2D array of size (KDE_RESOLUTION, KDE_RESOLUTION).

    Always uses foot-point coordinates (ground contact) so the heatmap
    stays on the floor plane and never bleeds onto buildings / sky.
    """
    res = KDE_RESOLUTION
    heatmap = np.zeros((res, res), dtype=np.float32)

    if not detections:
        return heatmap

    calibrated = calibration.is_calibrated()

    for d in detections:
        if calibrated and d.get("wx") is not None:
            # World coordinates → heatmap grid
            gx = int((d["wx"] / WORLD_GRID_W) * res)
            gy = int((d["wy"] / WORLD_GRID_H) * res)
        else:
            # Uncalibrated — use foot-point pixel coords normalised to frame
            from config import CAPTURE_W, CAPTURE_H
            gx = int((d["fx"] / CAPTURE_W) * res)
            gy = int((d["fy"] / CAPTURE_H) * res)

        if 0 <= gx < res and 0 <= gy < res:
            heatmap[gy, gx] += 1.0

    # Apply Gaussian Kernel (KDE Bandwidth)
    sigma = (KDE_BANDWIDTH / ((WORLD_GRID_W + WORLD_GRID_H) / 2)) * res
    k_size = int(sigma * 3) * 2 + 1

    if k_size > 1:
        heatmap = cv2.GaussianBlur(heatmap, (k_size, k_size), sigma)

    # Mask out static obstacles
    for obs_poly in OBSTACLE_POLYGONS:
        mask = np.zeros((res, res), dtype=np.uint8)
        poly_pts = (np.array(obs_poly.exterior.coords) * [res/WORLD_GRID_W, res/WORLD_GRID_H]).astype(np.int32)
        cv2.fillPoly(mask, [poly_pts], 1)
        heatmap[mask == 1] = 0

    # Normalize for visualization
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


# ─────────────────────────────────────────────────────────────────────────────
# Risk level from physical density
# ─────────────────────────────────────────────────────────────────────────────

def get_risk(density_per_m2):
    for threshold, label, color in DENSITY_THRESHOLDS:
        if density_per_m2 < threshold:
            return label, color
    return DENSITY_THRESHOLDS[-1][1], DENSITY_THRESHOLDS[-1][2]


# ─────────────────────────────────────────────────────────────────────────────
# Heatmap colour for a density value
# ─────────────────────────────────────────────────────────────────────────────

def get_grid_color(density):
    # Dynamically align heatmap colors with config thresholds
    for threshold, label, color in DENSITY_THRESHOLDS:
        if density < threshold:
            return color
    return DENSITY_THRESHOLDS[-1][2]


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell density  —  uses individual real areas when calibrated
# ─────────────────────────────────────────────────────────────────────────────

def compute_cell_densities(zone_counts):
    """
    Returns a [GRID_ROWS][GRID_COLS] table of people/m².

    When calibrated, each cell uses its own real floor area from
    calibration.cell_area_m2(r, c).  Far-field cells (small in pixels, large
    in reality) get a larger denominator and therefore lower density, which
    is the physically correct result.

    When uncalibrated, falls back to the uniform CELL_AREA_M2 constant.
    """
    return [
        [
            zone_counts[r][c] / max(calibration.cell_area_m2(r, c), 0.001)
            for c in range(GRID_COLS)
        ]
        for r in range(GRID_ROWS)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Density Tracker
# ─────────────────────────────────────────────────────────────────────────────

class DensityTracker:

    def __init__(self):
        self.count_history   = deque(maxlen=BUFFER_SIZE)
        self.density_history = deque(maxlen=BUFFER_SIZE)
        self.physical_density_history = deque(maxlen=BUFFER_SIZE)
        self.room_density_history = deque(maxlen=BUFFER_SIZE)
        self._last_hull_pts_px = None
        self._last_hull_pts_w  = None
        self._last_hull_m2     = 0.0

    def update(self, corrected_count, zone_counts, detections, frame_w, frame_h, cnn_count=None):
        """
        Update density tracker with latest detections and counts.
        
        ✓ ALL area calculations are based exclusively on hull of actual detections
        
        Args:
            corrected_count — occlusion-corrected person count (float)
            zone_counts     — 2-D list from detector
            detections      — list of dicts (must contain wx, wy or cx, cy)
            frame_w/h       — frame dimensions (for reference)

        Returns:
            stable_count    — int, smoothed count (from history buffer)
            smooth_density  — float, smoothed normalised density
            phys_density    — float, smoothed people per m2 (ignoring unused space)
            room_density    — float, smoothed people per m2 (total usable room area)
            cell_densities  — [GRID_ROWS][GRID_COLS] people/m² per cell
            hull_pts_px     — np.ndarray pixel coords of hull (for drawing)
            hull_area_m2    — float, crowd footprint area in m² 
            kde_map         — np.ndarray (res, res) normalized heatmap
                              (calculated from actual hull, NOT constants)
        """

        # ── Apply Manual ROI Filtering ───────────────────────────────────
        if ROI_POLYGON and calibration.is_calibrated():
            detections = [d for d in detections if ROI_POLYGON.contains(Point(d["wx"], d["wy"]))]
            corrected_count = float(len(detections))
            # Update zone_counts to match filtered detections
            zone_counts = [[0] * GRID_COLS for _ in range(GRID_ROWS)]
            for d in detections:
                r, c = calibration.world_to_grid(d["wx"], d["wy"])
                zone_counts[r][c] += 1

        # ── COUNT FUSION (YOLO + CNN) ────────────────────────────────────
        # If CNN count is provided, we fuse it based on confidence.
        # High density areas favor CNN; Low density favors YOLO.
        if cnn_count is not None:
            corrected_count = max(corrected_count, cnn_count)

        # ── HULL-BASED AREA CALCULATION ──────────────────────────────────
        # All area metrics derived exclusively from concave hull (alpha shape) of crowd
        # (When calibrated: perspective-correct world coordinates)
        # (When not calibrated: pixel hull scaled by monitored floor extent)
        hull_area_m2, hull_pts_px, hull_pts_w, hull_type, alpha_value = calculate_crowd_area_m2(detections)

        # ── KDE CALCULATION ──────────────────────────────────────────────
        kde_map = compute_kde_heatmap(detections)

        self._last_hull_pts_px = hull_pts_px
        self._last_hull_pts_w  = hull_pts_w
        self._last_hull_m2     = hull_area_m2

        # ── Smoothed count ────────────────────────────────────────────────
        self.count_history.append(corrected_count)

        stable_count = round(
            sum(self.count_history) / len(self.count_history)
        )

        # ── Physical density (People / Occupied Area) ─────────────────────
        # This value inherently ignores unused space
        current_phys_density = corrected_count / max(hull_area_m2, 0.1)
        self.physical_density_history.append(current_phys_density)
        
        smooth_phys_density = (
            sum(self.physical_density_history) / len(self.physical_density_history)
        )
        
        # ── Room Density (People / Total Usable Area) ─────────────────────
        current_room_density = corrected_count / TOTAL_USABLE_AREA
        self.room_density_history.append(current_room_density)
        smooth_room_density = (
            sum(self.room_density_history) / len(self.room_density_history)
        )

        # ── Place-aware normalised density trend ──────────────────────────
        place_density = get_normalized_density(corrected_count)
        self.density_history.append(place_density)

        smooth_density = (
            sum(self.density_history) / len(self.density_history)
        )

        # ── Per-cell density with real areas ─────────────────────────────
        cell_densities = compute_cell_densities(zone_counts)

        return (
            stable_count,
            smooth_density,
            smooth_phys_density,
            smooth_room_density,
            cell_densities,
            hull_pts_px,
            hull_area_m2,
            kde_map,
            hull_type,
            alpha_value
        )

    def history(self):
        return list(self.density_history)

    @property
    def last_hull_pts(self):
        return self._last_hull_pts_px

    @property
    def last_hull_pts_world(self):
        return self._last_hull_pts_w

    @property
    def last_hull_m2(self):
        return self._last_hull_m2
