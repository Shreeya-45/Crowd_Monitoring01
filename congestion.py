"""
Congestion Detector
===================
Grid-based congestion detection with perspective correction.

Divides the ROI bounding box into an NxM grid, counts people per cell,
and flags cells exceeding configurable thresholds with multi-level
severity (WARNING / CRITICAL).
"""

import numpy as np
import calibration
from config import WORLD_GRID_W, WORLD_GRID_H

class CongestionAlert:
    def __init__(self, cell_row: int, cell_col: int, cell_bounds: tuple, density_value: float, severity: str):
        self.cell_row = cell_row
        self.cell_col = cell_col
        self.cell_bounds = cell_bounds
        self.density_value = density_value
        self.severity = severity

class CongestionDetector:
    """Grid-based congestion detection with multi-level alerting."""

    def __init__(self, roi_polygon: np.ndarray, frame_shape: tuple,
                 rows: int = 8, cols: int = 8,
                 warning_threshold: float = 3.0,
                 critical_threshold: float = 6.0):
        self.rows = rows
        self.cols = cols
        self.warning_thresh = warning_threshold
        self.critical_thresh = critical_threshold
        self.frame_h, self.frame_w = frame_shape[:2]

        # Compute grid bounds from ROI polygon bounding box
        if roi_polygon is not None and len(roi_polygon) > 0:
            self.roi_x_min = int(np.min(roi_polygon[:, 0]))
            self.roi_y_min = int(np.min(roi_polygon[:, 1]))
            self.roi_x_max = int(np.max(roi_polygon[:, 0]))
            self.roi_y_max = int(np.max(roi_polygon[:, 1]))
        else:
            self.roi_x_min, self.roi_y_min = 0, 0
            self.roi_x_max, self.roi_y_max = self.frame_w, self.frame_h

        # Compute cell dimensions
        roi_w = max(self.roi_x_max - self.roi_x_min, 1)
        roi_h = max(self.roi_y_max - self.roi_y_min, 1)
        self.cell_w = roi_w / cols
        self.cell_h = roi_h / rows

        # Current density matrix
        self.density_matrix = np.zeros((rows, cols), dtype=np.float32)

    def get_cell(self, x: float, y: float) -> tuple:
        """Map a point to its grid cell (row, col)."""
        col = int((x - self.roi_x_min) / self.cell_w)
        row = int((y - self.roi_y_min) / self.cell_h)
        col = int(np.clip(col, 0, self.cols - 1))
        row = int(np.clip(row, 0, self.rows - 1))
        return row, col

    def get_cell_bounds(self, row: int, col: int) -> tuple:
        """Get pixel bounds (x1, y1, x2, y2) for a given cell."""
        x1 = int(self.roi_x_min + col * self.cell_w)
        y1 = int(self.roi_y_min + row * self.cell_h)
        x2 = int(x1 + self.cell_w)
        y2 = int(y1 + self.cell_h)
        return (x1, y1, x2, y2)

    def analyze(self, detections: list) -> tuple:
        """Analyze detections and return density matrix + congestion alerts.

        Args:
            detections: List of dicts containing assigned grid row and col.

        Returns:
            density_matrix: (rows, cols) array of per-cell counts.
            alerts: List of CongestionAlert for cells exceeding thresholds.
        """
        self.density_matrix = np.zeros((self.rows, self.cols), dtype=np.float32)
        calibrated = calibration.is_calibrated()

        # Count foot points per cell using pre-assigned row/col
        for det in detections:
            row = det['row']
            col = det['col']
            self.density_matrix[row, col] += 1.0

        # Generate alerts for cells exceeding thresholds
        alerts = []
        for r in range(self.rows):
            for c in range(self.cols):
                val = self.density_matrix[r, c]
                cell_bounds = self.get_cell_bounds(r, c)
                
                cell_poly = None
                if calibrated:
                    w_step = WORLD_GRID_W / self.cols
                    h_step = WORLD_GRID_H / self.rows
                    w_pts = np.array([
                        [c * w_step, r * h_step],
                        [(c + 1) * w_step, r * h_step],
                        [(c + 1) * w_step, (r + 1) * h_step],
                        [c * w_step, (r + 1) * h_step]
                    ], dtype=np.float32)
                    cell_poly = calibration.world_to_px(w_pts).tolist()

                if val >= self.critical_thresh:
                    alert = CongestionAlert(
                        cell_row=r, cell_col=c,
                        cell_bounds=cell_bounds,
                        density_value=val,
                        severity="CRITICAL"
                    )
                    alert.cell_poly = cell_poly
                    alerts.append(alert)
                elif val >= self.warning_thresh:
                    alert = CongestionAlert(
                        cell_row=r, cell_col=c,
                        cell_bounds=cell_bounds,
                        density_value=val,
                        severity="WARNING"
                    )
                    alert.cell_poly = cell_poly
                    alerts.append(alert)

        return self.density_matrix, alerts
