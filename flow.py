# flow.py — per-person motion vectors and per-cell net flow
# Inspired by Paper 1 (grid-based crowd flow prediction)
#
# Changes from original
# ─────────────────────
# • Centroid history stores  (wx, wy)  in metres when calibrated, so
#   velocity vectors dx/dy are in metres-per-frame (real-world speed).
#   Falls back to pixel deltas when uncalibrated (original behaviour).
#
# • draw_flow() uses  fx, fy  (pixel foot point) as the arrow origin
#   so arrows start at ground level rather than body centre — more
#   natural on an angled camera.
#
# • Cell-flow aggregation uses the detection's  row, col  which is now
#   computed in world space by detector.py when calibrated.

from collections import defaultdict
from config import FLOW_HISTORY_LEN, GRID_ROWS, GRID_COLS
import cv2
import calibration


class FlowTracker:
    """
    Stores position history per tracked ID.
    Emits per-person velocity vectors and per-cell net flow direction.

    When calibrated, positions and velocities are in metres.
    When uncalibrated, they are in pixels (original behaviour).
    """

    def __init__(self):
        # pid → [(pos_x, pos_y), ...]  — world metres or pixels
        self._history = defaultdict(list)

    def update(self, detections):
        """
        Args:
            detections: list of dicts from detector.run_detection()

        Returns:
            vectors   — list of (fx, fy, dx, dy, pid)
                          fx, fy  — pixel foot point (for drawing)
                          dx, dy  — velocity in world metres/frame
                                    (or pixels/frame if uncalibrated)
            cell_flow — 2-D list [row][col] of (net_dx, net_dy)
        """
        calibrated  = calibration.is_calibrated()
        active_pids = set()

        for d in detections:
            pid = d["pid"]
            if pid < 0:
                continue
            active_pids.add(pid)

            # Track in world coords when available, pixels otherwise
            if calibrated:
                pos = (d["wx"], d["wy"])
            else:
                pos = (d["cx"], d["cy"])

            self._history[pid].append(pos)
            if len(self._history[pid]) > FLOW_HISTORY_LEN + 1:
                self._history[pid].pop(0)

        # Purge stale IDs
        for pid in list(self._history):
            if pid not in active_pids:
                del self._history[pid]

        # Build velocity vectors
        vectors = []
        for d in detections:
            pid  = d["pid"]
            hist = self._history.get(pid, [])

            if len(hist) >= 2:
                dx = hist[-1][0] - hist[-2][0]
                dy = hist[-1][1] - hist[-2][1]
            else:
                dx, dy = 0.0, 0.0

            vectors.append((d["fx"], d["fy"], dx, dy, pid))

        # Aggregate net flow per grid cell
        cell_flow = [[(0.0, 0.0)] * GRID_COLS for _ in range(GRID_ROWS)]
        for d, (_, _, dx, dy, _pid) in zip(detections, vectors):
            r, c = d["row"], d["col"]
            fx_c, fy_c = cell_flow[r][c]
            cell_flow[r][c] = (fx_c + dx, fy_c + dy)

        return vectors, cell_flow


def draw_flow(frame, vectors, cell_flow, frame_h, frame_w):
    """
    Draw per-person motion arrows and per-cell aggregate flow arrow.

    When calibrated, dx/dy are in metres — we scale them to pixels for
    display using a fixed visual scale factor.
    When uncalibrated, dx/dy are already in pixels.

    All drawing is done in-place on frame.
    """
    calibrated = calibration.is_calibrated()

    # Visual scale: how many pixels to draw per unit of dx/dy
    # Calibrated:   1 m  → 40 px   (arrows stay readable at normal walking speed)
    # Uncalibrated: same as original (4 px per pixel-delta)
    px_per_unit = 40.0 if calibrated else 4.0

    cell_w = frame_w // GRID_COLS
    cell_h = frame_h // GRID_ROWS

    # Per-person arrows (thin, cyan)
    for (fx, fy, dx, dy, _pid) in vectors:
        magnitude = abs(dx) + abs(dy)
        # Calibrated: skip if < 0.05 m/frame (~1.0 m/s at ~20fps); uncal: < 2px
        threshold = 0.05 if calibrated else 2.0
        if magnitude < threshold:
            continue
        ex = int(fx + dx * px_per_unit)
        ey = int(fy + dy * px_per_unit)
        cv2.arrowedLine(frame, (int(fx), int(fy)), (ex, ey),
                        (0, 255, 255), 1, tipLength=0.4)

    # Per-cell net flow arrows (thicker, white)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            fdx, fdy = cell_flow[r][c]
            magnitude = abs(fdx) + abs(fdy)
            threshold = 0.05 if calibrated else 1.0
            if magnitude < threshold:
                continue
            ox = c * cell_w + cell_w // 2
            oy = r * cell_h + cell_h // 2
            ex = int(ox + fdx * px_per_unit * 0.75)
            ey = int(oy + fdy * px_per_unit * 0.75)
            cv2.arrowedLine(frame, (ox, oy), (ex, ey),
                            (255, 255, 255), 2, tipLength=0.35)
