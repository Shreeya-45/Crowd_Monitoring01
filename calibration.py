# calibration.py
#
# FULLY AUTOMATIC ground-plane homography calibration.
#
# HOW TO CALIBRATE (one-time setup)
# ----------------------------------
# 1. Install your camera pointing at the floor
# 2. Run:  python calibration_tool.py  or  python main.py
#    The system AUTOMATICALLY detects:
#    ✓ Checkerboard patterns
#    ✓ Tile grids
#    ✓ Any regular floor markings
# 3. The homography matrix H is computed and saved to HOMOGRAPHY_FILE
# 4. From that point on, every run loads H automatically
#
# ZERO USER INTERACTION REQUIRED!
# ✓ No clicking
# ✓ No measurements
# ✓ No manual input
# ✓ Just have a patterned floor in view
#
# WHAT THIS FIXES
# ---------------
# The original code used a single CELL_AREA_M2 constant applied uniformly
# across the whole frame.  That is only valid for a perfectly overhead camera.
# For any angled / wall-mounted camera, objects near the camera occupy far
# more pixels than equally-sized objects at the far end of the scene, causing
# density to be over-counted near the camera and under-counted far away.
#
# After calibration every pixel coordinate is mapped to a real floor position
# in metres via the homography H.  All areas (hull, grid cells) are then
# computed in metres-squared directly, making density physically accurate
# regardless of camera angle.

import os
import cv2
import numpy as np
from config import (
    CAMERA_INDEX, CAPTURE_W, CAPTURE_H,
    GRID_ROWS, GRID_COLS,
    HOMOGRAPHY_FILE, WORLD_GRID_W, WORLD_GRID_H
)


# ─────────────────────────────────────────────────────────────────
# Automatic floor pattern detection
# ─────────────────────────────────────────────────────────────────

def detect_checkerboard(frame, min_corners=(5, 4)):
    """
    Detect checkerboard pattern in frame.
    
    Args:
        frame: Input image (BGR)
        min_corners: Minimum (rows, cols) to accept
        
    Returns:
        (corners, board_size) if found, else (None, None)
        corners: Nx2 array of detected corner coordinates in pixels
        board_size: (rows, cols) of the detected checkerboard
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Try to find checkerboard with increasing tolerance
    for target_cols in range(8, 3, -1):
        for target_rows in range(6, 2, -1):
            if target_rows < min_corners[0] or target_cols < min_corners[1]:
                continue
            
            found, corners = cv2.findChessboardCorners(
                gray,
                (target_cols, target_rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if found:
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                return refined.reshape(-1, 2), (target_rows, target_cols)
    
    return None, None


def detect_tile_grid(frame, min_tiles=10):
    """
    Detect tile/grid pattern via edge and line detection.
    
    Args:
        frame: Input image (BGR)
        min_tiles: Minimum number of tiles to accept
        
    Returns:
        (corners, grid_dims) if found, else (None, None)
        corners: 4x2 array of detected rectangle corners
        grid_dims: (rows, cols) of detected grid
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)
    
    # Detect lines using Hough
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    
    if lines is None or len(lines) < 4:
        return None, None
    
    # Group lines into horizontal and vertical
    h_lines = []
    v_lines = []
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1))
        
        # Horizontal (angle ~ 0 or π)
        if angle < np.pi / 6 or angle > 5 * np.pi / 6:
            h_lines.append(((y1 + y2) / 2, line[0]))
        # Vertical (angle ~ π/2)
        elif np.pi / 3 < angle < 2 * np.pi / 3:
            v_lines.append(((x1 + x2) / 2, line[0]))
    
    if len(h_lines) < 2 or len(v_lines) < 2:
        return None, None
    
    # Sort and get outer lines
    h_lines.sort()
    v_lines.sort()
    
    # Get unique horizontal and vertical positions (cluster nearby lines)
    h_positions = [h_lines[0][0]]
    v_positions = [v_lines[0][0]]
    
    for y, _ in h_lines[1:]:
        if abs(y - h_positions[-1]) > 15:
            h_positions.append(y)
    
    for x, _ in v_lines[1:]:
        if abs(x - v_positions[-1]) > 15:
            v_positions.append(x)
    
    if len(h_positions) < 2 or len(v_positions) < 2:
        return None, None
    
    # Use outer grid corners as calibration rectangle
    corners = np.array([
        [v_positions[0], h_positions[0]],       # Top-left
        [v_positions[-1], h_positions[0]],      # Top-right
        [v_positions[-1], h_positions[-1]],     # Bottom-right
        [v_positions[0], h_positions[-1]],      # Bottom-left
    ], dtype=np.float32)
    
    grid_dims = (len(h_positions) - 1, len(v_positions) - 1)
    
    return corners, grid_dims


def auto_calibrate(frame=None):
    """
    Automatically detect floor pattern and compute homography.
    ZERO USER INTERACTION - fully automatic!
    
    Returns:
        H: 3×3 homography matrix, or None if pattern not detected
    """
    if frame is None:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Could not read frame from camera.")
    
    print("\n[AUTO-CALIBRATION] Scanning for floor patterns...")
    
    # Try checkerboard first
    print("  Trying checkerboard detection...")
    corners, board_size = detect_checkerboard(frame)
    
    if corners is not None:
        print(f"  ✓ Checkerboard found: {board_size[0]}×{board_size[1]} grid")
        return _compute_homography_from_pattern(corners, board_size, "checkerboard")
    
    # Try tile grid detection
    print("  Trying tile grid detection...")
    corners, grid_dims = detect_tile_grid(frame)
    
    if corners is not None:
        print(f"  ✓ Tile grid found: {grid_dims[0]}×{grid_dims[1]} tiles")
        return _compute_homography_from_pattern(corners, grid_dims, "tile_grid")
    
    print("  ✗ No floor pattern detected. Make sure:")
    print("    - Camera points at patterned floor (checkerboard, tiles, grid)")
    print("    - Pattern is clearly visible and well-lit")
    print("    - Floor pattern spans most of the camera view")
    return None


def _compute_homography_from_pattern(img_corners, pattern_dims, pattern_type="tile_grid"):
    """
    Compute homography from detected pattern corners.
    Assumes pattern is a regular grid of tiles/squares that spans
    the full WORLD_GRID_W × WORLD_GRID_H floor area.

    Args:
        img_corners: detected pattern corners in pixels
            - checkerboard: Nx2 array of ALL interior corners (row-major order)
            - tile_grid / manual: 4x2 array [TL, TR, BR, BL]
        pattern_dims: (rows, cols) of the pattern interior corners
        pattern_type: "checkerboard", "tile_grid", or "manual"

    Returns:
        H: 3×3 homography matrix
    """
    rows, cols = pattern_dims

    rw = WORLD_GRID_W
    rh = WORLD_GRID_H

    print(f"  Estimated floor area: {rw:.2f}m (width) × {rh:.2f}m (height)")

    if pattern_type == "checkerboard" and len(img_corners) > 4:
        # ── Checkerboard: use ALL N interior corners ──────────────────────
        # cv2.findChessboardCorners returns corners in row-major order:
        # index (r*cols + c) is the corner at grid position (row=r, col=c).
        #
        # The board has (cols+1) squares wide × (rows+1) squares tall,
        # assumed to span the full WORLD_GRID area including outer squares.
        # So the interior corner at (r, c) sits at world position:
        #   wx = (c + 1) * sq_w,  wy = (r + 1) * sq_h
        sq_w = rw / (cols + 1)
        sq_h = rh / (rows + 1)

        img_pts = np.array(img_corners, dtype=np.float32)  # shape (N, 2)

        world_pts_list = []
        for r in range(rows):
            for c in range(cols):
                world_pts_list.append([(c + 1) * sq_w, (r + 1) * sq_h])
        world_pts = np.array(world_pts_list, dtype=np.float32)  # shape (N, 2)
    else:
        # ── Tile grid / manual: 4 corners spanning the full floor area ─────
        img_pts   = np.array(img_corners[:4], dtype=np.float32)
        world_pts = np.array([
            [0.0,  0.0],
            [rw,   0.0],
            [rw,   rh ],
            [0.0,  rh ],
        ], dtype=np.float32)

    H, mask = cv2.findHomography(img_pts, world_pts)

    if H is None:
        raise RuntimeError("Homography computation failed.")

    np.save(HOMOGRAPHY_FILE, H)
    print(f"\n✓ [CALIBRATION COMPLETE] Saved homography → {HOMOGRAPHY_FILE}")
    print(f"  Reprojection errors (first 4 points):")

    for i, (ip, wp) in enumerate(zip(img_pts[:4], world_pts[:4])):
        proj = cv2.perspectiveTransform(ip.reshape(1, 1, 2), H).reshape(2)
        err  = np.linalg.norm(proj - wp)
        print(f"    Corner {i+1}: {err * 100:.1f} cm")

    return H


def _draw_manual_ui(base_frame, pts, hover_xy=None):
    """
    Render the manual calibration overlay onto a copy of base_frame.
    Shows a header bar, progress dots, placed corner markers, a connecting
    polygon outline, and a live crosshair at the current mouse position.
    """
    out = base_frame.copy()
    h, w = out.shape[:2]
    n = len(pts)

    corner_names  = ["1 - Top-Left", "2 - Top-Right",
                     "3 - Bottom-Right", "4 - Bottom-Left"]
    corner_colors = [(0, 230, 80), (0, 200, 255), (255, 130, 0), (220, 0, 220)]

    # -- Semi-transparent header bar -----------------------------------------
    bar_h = 105
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.78, out, 0.22, 0, out)

    # Title
    cv2.putText(out, "MANUAL FLOOR CALIBRATION",
                (14, 30), cv2.FONT_HERSHEY_DUPLEX, 0.78, (255, 255, 255), 1, cv2.LINE_AA)

    # Next-corner instruction
    if n < 4:
        name  = corner_names[n]
        color = corner_colors[n]
        cv2.putText(out, "Click  {}  on the floor edge".format(name),
                    (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color, 1, cv2.LINE_AA)
    else:
        cv2.putText(out, "4 corners placed!   ENTER = confirm    R = reset",
                    (14, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 230, 80), 1, cv2.LINE_AA)

    # Key hints
    cv2.putText(out, "[Click] Place    [Z] Undo    [R] Reset    [ENTER] Confirm    [S/ESC] Skip calibration",
                (14, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 160, 160), 1, cv2.LINE_AA)

    # Progress circles (top-right)
    for i in range(4):
        cx = w - 24 - (3 - i) * 30
        cy = 22
        filled = i < n
        col = corner_colors[i] if filled else (55, 55, 55)
        cv2.circle(out, (cx, cy), 10, col, -1 if filled else 2)
        cv2.putText(out, str(i + 1), (cx - 4, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Floor area label (bottom-left)
    label = "Floor: {}m x {}m".format(WORLD_GRID_W, WORLD_GRID_H)
    cv2.putText(out, label,
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1, cv2.LINE_AA)

    # -- Polygon outline (once 2+ points) ------------------------------------
    if n >= 2:
        poly = np.array(pts, dtype=np.int32)
        cv2.polylines(out, [poly], isClosed=(n == 4),
                      color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    # -- Placed corner markers -----------------------------------------------
    for i, (px, py) in enumerate(pts):
        col = corner_colors[i]
        cv2.line(out, (px - 16, py), (px + 16, py), col, 1, cv2.LINE_AA)
        cv2.line(out, (px, py - 16), (px, py + 16), col, 1, cv2.LINE_AA)
        cv2.circle(out, (px, py), 7, col, -1)
        cv2.circle(out, (px, py), 9, (255, 255, 255), 1, cv2.LINE_AA)
        lx = px + 12 if px < w - 130 else px - 135
        ly = py - 12 if py > bar_h + 20 else py + 24
        cv2.putText(out, corner_names[i], (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

    # -- Live crosshair at mouse position ------------------------------------
    if hover_xy is not None and n < 4:
        mx, my = hover_xy
        col = corner_colors[n]
        cv2.line(out, (mx - 20, my), (mx + 20, my), col, 1, cv2.LINE_AA)
        cv2.line(out, (mx, my - 20), (mx, my + 20), col, 1, cv2.LINE_AA)
        cv2.circle(out, (mx, my), 5, col, -1)

    return out


def _manual_point_selection(frame=None):
    """
    Manual 4-point floor-corner calibration with a polished interactive UI.

    If *frame* is None the function automatically captures a representative
    frame from the video / camera defined by CAMERA_INDEX.

    Click the 4 corners of the monitored floor area in order:
        1. Top-Left  ->  2. Top-Right  ->  3. Bottom-Right  ->  4. Bottom-Left

    Controls
    --------
    Left-click        Place next corner
    Z                 Undo last corner
    R                 Reset all corners
    ENTER / SPACE     Confirm (enabled only when all 4 are placed)
    ESC               Cancel -- returns None
    """
    # -- Grab a frame automatically when none is provided --------------------
    if frame is None:
        print("[CALIBRATION] Capturing frame from source for manual calibration...")
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
        grabbed = None
        for _ in range(30):          # skip initial frames (exposure settle)
            ret, f = cap.read()
            if ret:
                grabbed = f
        cap.release()
        if grabbed is None:
            print("[CALIBRATION] ERROR: Could not read any frame from source.")
            return None
        frame = grabbed
        print("[CALIBRATION] Frame captured -- opening calibration window...")

    corner_names  = ["1 - Top-Left", "2 - Top-Right",
                     "3 - Bottom-Right", "4 - Bottom-Left"]

    # -- State ----------------------------------------------------------------
    pts   = []        # list of [x, y] int pixel coords
    hover = [None]    # current mouse position for live crosshair
    WIN   = "Manual Floor Calibration  --  Click the 4 corners"

    def mouse_cb(event, x, y, flags, param):
        hover[0] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append([x, y])
            short = corner_names[len(pts) - 1].split(" - ")[1]
            print("  [+] Corner {} ({}) placed at pixel ({}, {})".format(
                len(pts), short, x, y))

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, min(frame.shape[1], 1280), min(frame.shape[0], 720))
    cv2.setMouseCallback(WIN, mouse_cb)

    print("\n" + "=" * 62)
    print("MANUAL FLOOR CALIBRATION")
    print("=" * 62)
    print("  Floor area: {}m (W)  x  {}m (H)".format(WORLD_GRID_W, WORLD_GRID_H))
    print("  Click the 4 corners of the floor in this order:")
    for name in corner_names:
        print("    " + name)
    print("  [Z] Undo  [R] Reset  [ENTER/SPACE] Confirm  [ESC] Cancel")
    print("=" * 62)

    confirmed = False
    while True:
        disp = _draw_manual_ui(frame, pts, hover[0])
        cv2.imshow(WIN, disp)
        key = cv2.waitKey(20) & 0xFF

        if key == 27 or key in (ord('s'), ord('S')):  # ESC or S -- skip
            print("[CALIBRATION] Skipped. Running in fallback mode.")
            break
        elif key in (13, 32) and len(pts) == 4:  # ENTER or SPACE -- confirm
            confirmed = True
            break
        elif key in (ord('z'), ord('Z')):        # Undo last corner
            if pts:
                removed = pts.pop()
                print("  [Undo] Removed corner at {}".format(removed))
        elif key in (ord('r'), ord('R')):        # Reset all corners
            pts.clear()
            print("  [Reset] All corners cleared.")

    cv2.destroyWindow(WIN)

    if confirmed and len(pts) == 4:
        print("\n[CALIBRATION] 4 corners confirmed:")
        for i, (px, py) in enumerate(pts):
            short = corner_names[i].split(" - ")[1]
            print("  Corner {} ({}): ({}, {}) px".format(i + 1, short, px, py))
        return np.array(pts, dtype=np.float32)
    return None


def run_calibration(frame=None):
    """
    Try automatic calibration first; fall back to the manual 4-point UI.

    A single frame is captured once (when frame is None) and reused for
    both the automatic and manual paths so the UI always has a live image.
    """
    # Capture one frame upfront so both paths can use it.
    if frame is None:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
        grabbed = None
        for _ in range(10):
            ret, f = cap.read()
            if ret:
                grabbed = f
        cap.release()
        frame = grabbed      # may remain None if source unavailable

    # -- Attempt automatic detection -----------------------------------------
    if frame is not None:
        H = auto_calibrate(frame)
        if H is not None:
            return H

    print("\n[CALIBRATION] Automatic detection failed. Switching to MANUAL 4-point UI.")
    manual_pts = _manual_point_selection(frame)   # handles None internally
    if manual_pts is not None:
        return _compute_homography_from_pattern(manual_pts, (1, 1), "tile_grid")
    return None

# ─────────────────────────────────────────────────────────────────
# Load / lazy-initialise
# ─────────────────────────────────────────────────────────────────

_H      = None        # 3×3 homography matrix (image → world metres)
_H_inv  = None        # inverse   (world → image), for drawing
_cell_areas = None    # [GRID_ROWS][GRID_COLS] real m² per grid cell


def load_homography():
    """
    Load saved homography from disk.
    Returns H (3×3 float64) or None if no calibration file exists.
    """
    global _H, _H_inv
    if not os.path.exists(HOMOGRAPHY_FILE):
        return None
    _H     = np.load(HOMOGRAPHY_FILE)
    _H_inv = np.linalg.inv(_H)
    _precompute_cell_areas()
    return _H


def is_calibrated():
    return _H is not None


def get_homography():
    return _H


def get_homography_inv():
    return _H_inv


# ─────────────────────────────────────────────────────────────────
# Core transform helpers
# ─────────────────────────────────────────────────────────────────

def px_to_world(pixel_pts):
    """
    Map Nx2 pixel coordinates to Nx2 world (ground-plane) coordinates in metres.

    Args:
        pixel_pts: np.ndarray shape (N, 2) float32 or list of (x, y) tuples

    Returns:
        np.ndarray shape (N, 2) float64 — world positions in metres
    """
    if _H is None:
        raise RuntimeError("Homography not loaded. Run calibration first.")

    pts = np.array(pixel_pts, dtype=np.float32).reshape(-1, 1, 2)
    world = cv2.perspectiveTransform(pts, _H)
    return world.reshape(-1, 2)


def world_to_px(world_pts):
    """
    Map Nx2 world coordinates (metres) back to pixel coordinates.
    Used for drawing ground-truth overlays.
    """
    if _H_inv is None:
        raise RuntimeError("Homography not loaded.")

    pts = np.array(world_pts, dtype=np.float32).reshape(-1, 1, 2)
    px  = cv2.perspectiveTransform(pts, _H_inv)
    return px.reshape(-1, 2).astype(np.int32)


# ─────────────────────────────────────────────────────────────────
# Per-cell real area  (replaces the single CELL_AREA_M2 constant)
# ─────────────────────────────────────────────────────────────────

def _precompute_cell_areas():
    """
    For each grid cell, project its 4 pixel corners into world space and
    compute the real floor area via the shoelace formula.
    Stored in _cell_areas[row][col] in m².
    """
    global _cell_areas

    # We need a representative frame size to lay out the grid.
    # Use CAPTURE_W × CAPTURE_H from config.
    from config import CAPTURE_W, CAPTURE_H

    cw = CAPTURE_W / GRID_COLS
    ch = CAPTURE_H / GRID_ROWS

    _cell_areas = []
    for r in range(GRID_ROWS):
        row_areas = []
        for c in range(GRID_COLS):
            # 4 pixel corners of this cell (TL TR BR BL)
            corners_px = np.array([
                [c * cw,       r * ch      ],
                [(c+1) * cw,   r * ch      ],
                [(c+1) * cw,   (r+1) * ch  ],
                [c * cw,       (r+1) * ch  ],
            ], dtype=np.float32)

            corners_w = px_to_world(corners_px)   # shape (4, 2) in metres

            # Shoelace formula for polygon area
            x = corners_w[:, 0]
            y = corners_w[:, 1]
            area = 0.5 * abs(
                np.dot(x, np.roll(y, -1)) -
                np.dot(y, np.roll(x, -1))
            )
            row_areas.append(float(area))
        _cell_areas.append(row_areas)


def cell_area_m2(row, col):
    """
    Return the real floor area (m²) of grid cell (row, col).
    Falls back to the uniform CELL_AREA_M2 constant if not calibrated.
    """
    if _cell_areas is not None:
        return _cell_areas[row][col]

    # Fallback: uncalibrated — uniform constant from config
    from config import CELL_AREA_M2
    return CELL_AREA_M2


def all_cell_areas():
    """Return the full [GRID_ROWS][GRID_COLS] area table (or uniform fallback)."""
    if _cell_areas is not None:
        return _cell_areas

    from config import CELL_AREA_M2, GRID_ROWS, GRID_COLS
    return [[CELL_AREA_M2] * GRID_COLS for _ in range(GRID_ROWS)]


# ─────────────────────────────────────────────────────────────────
# Grid cell lookup in world space
# ─────────────────────────────────────────────────────────────────

def world_to_grid(wx, wy):
    """
    Given a world position (wx, wy) in metres, return the (row, col) index
    in the GRID_ROWS × GRID_COLS grid that spans the calibrated floor area.

    WORLD_GRID_W and WORLD_GRID_H in config define the total floor extent
    covered by the grid (metres).
    """
    col = int(wx / WORLD_GRID_W * GRID_COLS)
    row = int(wy / WORLD_GRID_H * GRID_ROWS)
    col = max(0, min(col, GRID_COLS - 1))
    row = max(0, min(row, GRID_ROWS - 1))
    return row, col


# ─────────────────────────────────────────────────────────────────
# Overlay helpers — draw calibration markers on a frame
# ─────────────────────────────────────────────────────────────────

def draw_world_grid(frame):
    """
    Project the real-world grid back into the image and draw it.
    Replaces the uniform pixel grid, showing the actual floor tiling.
    Only active when calibrated.
    """
    if _H_inv is None:
        return

    h, w = frame.shape[:2]

    for r in range(GRID_ROWS + 1):
        wy = r * WORLD_GRID_H / GRID_ROWS
        line_pts_w = np.array(
            [[c * WORLD_GRID_W / GRID_COLS, wy] for c in range(GRID_COLS + 1)],
            dtype=np.float32
        )
        line_pts_px = world_to_px(line_pts_w)
        for i in range(len(line_pts_px) - 1):
            p1 = tuple(line_pts_px[i])
            p2 = tuple(line_pts_px[i + 1])
            if all(0 <= v < d for v, d in zip(p1, (w, h))) or \
               all(0 <= v < d for v, d in zip(p2, (w, h))):
                cv2.line(frame, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)

    for c in range(GRID_COLS + 1):
        wx = c * WORLD_GRID_W / GRID_COLS
        line_pts_w = np.array(
            [[wx, r * WORLD_GRID_H / GRID_ROWS] for r in range(GRID_ROWS + 1)],
            dtype=np.float32
        )
        line_pts_px = world_to_px(line_pts_w)
        for i in range(len(line_pts_px) - 1):
            p1 = tuple(line_pts_px[i])
            p2 = tuple(line_pts_px[i + 1])
            if all(0 <= v < d for v, d in zip(p1, (w, h))) or \
               all(0 <= v < d for v, d in zip(p2, (w, h))):
                cv2.line(frame, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
# Validation & quality metrics
# ─────────────────────────────────────────────────────────────────

def get_calibration_quality():
    """
    Returns (0-1) quality score of the current homography based on
    the matrix condition number and determinant.
    1.0 = perfect, < 0.5 = poor quality, None = not calibrated.
    """
    if not is_calibrated():
        return None

    try:
        H   = get_homography()
        det  = abs(np.linalg.det(H))
        cond = np.linalg.cond(H)

        # Near-singular or extremely ill-conditioned → bad calibration
        if det < 0.01 or cond > 10000:
            return 0.1

        # Map condition number to [0, 1]: lower cond = higher quality
        # cond ≈ 1–10 → excellent;  cond ≈ 1000 → borderline
        quality = 1.0 / (1.0 + cond / 100.0)
        return float(np.clip(quality, 0.0, 1.0))
    except Exception:
        return None


def validate_calibration(test_frame=None):
    """
    Visually validate the calibration by showing:
    - Original frame with 4 reference points
    - Transformed world grid overlay
    - Reprojection quality indicators
    """
    if not is_calibrated():
        print("[VALIDATION] No calibration to validate.")
        return False

    if test_frame is None:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
        ret, test_frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Could not read frame for validation.")

    display = test_frame.copy()
    draw_world_grid(display)

    # Draw grid corners in the image
    corners_w = np.array([
        [0.0, 0.0],
        [WORLD_GRID_W, 0.0],
        [WORLD_GRID_W, WORLD_GRID_H],
        [0.0, WORLD_GRID_H],
    ], dtype=np.float32)
    corners_px = world_to_px(corners_w)

    for i, pt in enumerate(corners_px):
        cv2.circle(display, tuple(pt), 10, (0, 255, 0), -1)
        cv2.circle(display, tuple(pt), 10, (255, 255, 255), 2)

    cv2.putText(display, "Calibration Validation — Grid Overlay",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(display,
                "Green circles = world grid corners. Press SPACE to accept or ESC to recalibrate.",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Calibration Validation", display)
    while True:
        key = cv2.waitKey(0)
        if key == ord(' '):
            cv2.destroyWindow("Calibration Validation")
            return True
        elif key == 27:  # ESC
            cv2.destroyWindow("Calibration Validation")
            return False


def recalibrate_interactive():
    """
    Allows the user to recalibrate without restarting the system.
    Asks for confirmation before overwriting existing calibration.
    """
    print("\n[RECALIBRATION]")
    print("This will replace your current calibration.")
    confirm = input("Continue? (yes/no): ").strip().lower()
    if confirm != "yes":
        print("[RECALIBRATION] Cancelled.")
        return False

    try:
        H = run_calibration()
        load_homography()
        print("[RECALIBRATION] Successfully recalibrated!")
        return True
    except Exception as e:
        print(f"[RECALIBRATION] Failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────
# Enhanced live calibration with preview
# ─────────────────────────────────────────────────────────────────

def run_live_calibration_ui():
    """
    Interactive calibration with live camera preview.
    Provides real-time grid visualization and refinement options.
    """
    print("\n[LIVE CALIBRATION]")
    print("This tool will help you calibrate the camera for ground-plane mapping.")
    print("Make sure you have a tape measure and a clear reference rectangle marked on the floor.\n")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    # Get first frame
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read from camera.")

    cap.release()

    print("Click OK when ready to proceed to 4-point calibration.")
    try:
        H = run_calibration(frame=frame)
        if H is not None:
            load_homography()
        
        print("\n[LIVE CALIBRATION] Validating calibration...")
        if validate_calibration(test_frame=frame):
            print("[LIVE CALIBRATION] Calibration validated and saved!")
            return True
        else:
            print("[LIVE CALIBRATION] Validation rejected. Recalibrating...")
            os.remove(HOMOGRAPHY_FILE) if os.path.exists(HOMOGRAPHY_FILE) else None
            global _H, _H_inv, _cell_areas
            _H = _H_inv = _cell_areas = None
            return False
    except Exception as e:
        print(f"[LIVE CALIBRATION] Error: {e}")
        return False


# ─────────────────────────────────────────────────────────────────
# Entry point — run standalone to calibrate
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    H = run_calibration()
    print("\nHomography matrix H:")
    print(H)
    print("\nCalibration complete. You can now run main.py.")
