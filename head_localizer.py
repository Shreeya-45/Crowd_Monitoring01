"""Foot-point localisation from bounding-box detections.

Extracts ground-contact positions (bottom-centre of bounding boxes) for use
in spatial density estimation. Bottom-centre is more spatially accurate than
box centre for overhead / angled camera views.
"""

from typing import Optional, List, Dict
import numpy as np

class HeadLocalizer:
    """Extract ground-contact positions from bounding-box detections.

    The *foot point* is the bottom-centre of each bounding box — the
    approximate ground-contact position of a standing person. An optional
    binary ROI mask can be supplied to filter out detections that fall
    outside the region of interest.
    """

    def __init__(self, roi_mask: Optional[np.ndarray] = None) -> None:
        self._roi_mask: Optional[np.ndarray] = None
        if roi_mask is not None:
            self.set_roi_mask(roi_mask)

    def set_roi_mask(self, mask: np.ndarray) -> None:
        """Update the ROI mask used for spatial filtering."""
        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError(f"ROI mask must be 2-D, got {mask.ndim}-D")
        self._roi_mask = mask

    @staticmethod
    def compute_foot_point(x1: float, y1: float, x2: float, y2: float) -> tuple:
        """Compute the foot point for a single bounding box."""
        head_x = (x1 + x2) / 2.0
        head_y = float(y2)  # bottom edge (ground contact)
        bbox_height = float(y2 - y1)
        return head_x, head_y, bbox_height

    def extract(self, detections: List[Dict]) -> List[Dict]:
        """Filter and update detections with foot points."""
        valid_detections = []
        for det in detections:
            x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
            fx, fy, bbox_h = self.compute_foot_point(x1, y1, x2, y2)
            
            # ROI filter
            if self._roi_mask is not None:
                h, w = self._roi_mask.shape
                px = int(np.clip(round(fx), 0, w - 1))
                py = int(np.clip(round(fy), 0, h - 1))
                if self._roi_mask[py, px] == 0:
                    continue  # outside ROI — skip

            det['fx'] = fx
            det['fy'] = fy
            det['bbox_h'] = bbox_h
            valid_detections.append(det)

        return valid_detections
