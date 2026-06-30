

# ui.py — HD Surveillance UI  (drop-in replacement)

import cv2
import numpy as np
import datetime
from density import get_grid_color, get_risk
from config import GRID_ROWS, GRID_COLS, DISPLAY_W, DISPLAY_H, STATIC_OBSTACLES, MANUAL_ROI, WORLD_GRID_W, WORLD_GRID_H
import calibration
import tkinter as tk
from context_risk import set_place, get_place

# ─────────────────────────────────────────────────────────────────────────────
# Place selection window  (upgraded with themed tk styles)
# ─────────────────────────────────────────────────────────────────────────────

def select_place():
    root = tk.Tk()
    root.title("CrowdSense — Select Monitoring Zone")
    root.geometry("380x200")
    root.resizable(False, False)
    root.configure(bg="#0d1117")

    tk.Label(root, text="SELECT MONITORING ZONE",
             bg="#0d1117", fg="#22d3ee",
             font=("Courier", 11, "bold")).pack(pady=(18, 4))
    tk.Label(root, text="Choose the environment type for context-aware risk scoring",
             bg="#0d1117", fg="#64748b",
             font=("Courier", 8)).pack(pady=(0, 10))

    places = ["School", "Railway Station", "Mall", "Hospital", "Stadium", "Office"]
    selected = tk.StringVar(value="School")

    frm = tk.Frame(root, bg="#131920", bd=1, relief="solid")
    frm.pack(fill="x", padx=20)
    menu = tk.OptionMenu(frm, selected, *places)
    menu.config(bg="#131920", fg="#f1e8e6", activebackground="#22d3ee",
                activeforeground="#0d1117", font=("Courier", 10),
                borderwidth=0, highlightthickness=0)
    menu["menu"].config(bg="#131920", fg="#f1e8e6",
                        activebackground="#22d3ee", activeforeground="#0d1117",
                        font=("Courier", 10))
    menu.pack(fill="x")

    def start():
        set_place(selected.get())
        root.destroy()

    tk.Button(root, text="▶  BEGIN MONITORING",
              command=start,
              bg="#22d3ee", fg="#0d1117",
              font=("Courier", 10, "bold"),
              activebackground="#4ade80",
              relief="flat", bd=0,
              padx=20, pady=8).pack(pady=16)

    root.mainloop()


# ─────────────────────────────────────────────────────────────────────────────
# Color palette  (BGR)
# ─────────────────────────────────────────────────────────────────────────────

C_BG        = ( 10,  12,  16)   # #100c0a
C_PANEL     = ( 13,  17,  23)   # #0d1117
C_PANEL2    = ( 18,  22,  30)   # slightly lighter panel
C_BORDER    = ( 40,  50,  65)   # refined border
C_BORDER2   = ( 28,  36,  48)   # subtle inner border
C_CARD      = ( 20,  26,  36)   # stat card bg
C_CARD2     = ( 24,  31,  43)   # hover-like lighter card

C_TEXT      = (230, 232, 241)   # primary text
C_MUTED     = ( 90, 110, 130)   # secondary text
C_DIM       = ( 55,  68,  80)   # very muted

C_CYAN      = (238, 211,  34)   # #22d3ee
C_CYAN_DIM  = (120, 105,  17)   # dim cyan for glow base
C_GREEN     = (128, 222,  74)   # #4ade80
C_YELLOW    = ( 36, 191, 251)   # #fbbf24
C_ORANGE    = ( 22, 151, 249)   # #f97316
C_RED       = ( 68,  68, 239)   # #ef4444
C_PURPLE    = (250, 139, 167)   # #a78bfa
C_BLUE      = (250, 165,  96)   # #60a5fa
C_EXCL      = ( 42,  30,  72)   # Deep purple for excluded space
C_TEAL      = (180, 200,  60)   # teal accent

RISK_COLORS = {
    "VERY LOW":  C_GREEN,
    "LOW":       C_GREEN,
    "MODERATE":  C_YELLOW,
    "HIGH":      C_ORANGE,
    "CRITICAL":  C_RED,
}

LOS_WIDTHS = {
    "VERY LOW":  0.12,
    "LOW":       0.28,
    "MODERATE":  0.46,
    "HIGH":      0.70,
    "CRITICAL":  0.96,
}


# ─────────────────────────────────────────────────────────────────────────────
# Low-level drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def filled_rect(img, pt1, pt2, color, alpha=1.0):
    x1, y1 = max(pt1[0], 0), max(pt1[1], 0)
    x2, y2 = min(pt2[0], img.shape[1]-1), min(pt2[1], img.shape[0]-1)
    if x1 >= x2 or y1 >= y2:
        return
    if alpha >= 1.0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    else:
        roi = img[y1:y2, x1:x2]
        ov  = roi.copy()
        cv2.rectangle(ov, (0, 0), (x2-x1, y2-y1), color, -1)
        cv2.addWeighted(ov, alpha, roi, 1-alpha, 0, roi)
        img[y1:y2, x1:x2] = roi


def border_rect(img, pt1, pt2, color, thickness=1, radius=0):
    if radius <= 0:
        cv2.rectangle(img, pt1, pt2, color, thickness)
    else:
        # Simple rounded rect via polylines
        x1, y1, x2, y2 = pt1[0], pt1[1], pt2[0], pt2[1]
        r = min(radius, (x2-x1)//2, (y2-y1)//2)
        pts = []
        for cx, cy, a0, a1 in [
            (x1+r, y1+r, 180, 270),
            (x2-r, y1+r,  270, 360),
            (x2-r, y2-r,    0,  90),
            (x1+r, y2-r,   90, 180),
        ]:
            for a in range(a0, a1+1, 5):
                rad = np.radians(a)
                pts.append([int(cx + r*np.cos(rad)), int(cy + r*np.sin(rad))])
        pts = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)


def hline(img, x1, x2, y, color=C_BORDER, t=1):
    if 0 <= y < img.shape[0]:
        cv2.line(img, (max(x1,0), y), (min(x2, img.shape[1]-1), y), color, t)


def vline(img, x, y1, y2, color=C_BORDER, t=1):
    if 0 <= x < img.shape[1]:
        cv2.line(img, (x, max(y1,0)), (x, min(y2, img.shape[0]-1)), color, t)


def put(img, text, pos, scale=0.45, color=C_TEXT, weight=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, weight, cv2.LINE_AA)


def put_bold(img, text, pos, scale=0.45, color=C_TEXT):
    """Simulates bolder text with a subtle shadow then main draw."""
    shadow = tuple(max(c - 60, 0) for c in color)
    cv2.putText(img, text, (pos[0]+1, pos[1]+1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, shadow, 2, cv2.LINE_AA)
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def text_size(text, scale=0.45, weight=1):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, weight)
    return w, h


def centered_text(img, text, cx, cy, scale=0.45, color=C_TEXT, weight=1):
    tw, th = text_size(text, scale, weight)
    put(img, text, (cx - tw//2, cy + th//2), scale, color, weight)


def centered_text_bold(img, text, cx, cy, scale=0.45, color=C_TEXT):
    tw, th = text_size(text, scale, 2)
    put_bold(img, text, (cx - tw//2, cy + th//2), scale, color)


def glow_circle(img, center, radius, color, alpha=0.35, layers=3):
    """Multi-layer soft glow behind a circle."""
    for i in range(layers, 0, -1):
        r = radius + i * 3
        a = alpha / (i * 1.5)
        ov = img.copy()
        cv2.circle(ov, center, r, color, -1)
        cv2.addWeighted(ov, a, img, 1-a, 0, img)


def draw_gradient_bar(img, x1, y1, x2, y2, col_left, col_right):
    """Horizontal gradient bar."""
    w = x2 - x1
    if w <= 0:
        return
    for i in range(w):
        t = i / max(w - 1, 1)
        c = tuple(int(col_left[j] * (1-t) + col_right[j] * t) for j in range(3))
        cv2.line(img, (x1+i, y1), (x1+i, y2), c, 1)


def section_header(img, text, px, y, pw):
    """Styled section label with accent line - balanced visibility."""
    # Background for section header
    filled_rect(img, (px, y - 2), (px + pw, y + 14), C_CARD2, alpha=0.4)
    put(img, text, (px + 10, y + 9), 0.38, C_CYAN, 1)  # Lighter weight
    tw, _ = text_size(text, 0.38, 1)
    hline(img, px + 10 + tw + 8, px + pw - 10, y + 5, C_CYAN, 1)  # Thinner line


# ─────────────────────────────────────────────────────────────────────────────
# Feed area drawings
# ─────────────────────────────────────────────────────────────────────────────

def draw_grid_heatmap(frame, cell_densities):
    h, w = frame.shape[:2]
    calibrated = calibration.is_calibrated()
    overlay = frame.copy()

    if calibrated:
        w_step = WORLD_GRID_W / GRID_COLS
        h_step = WORLD_GRID_H / GRID_ROWS

        # 1. Fill cells on overlay
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                d = cell_densities[r][c]
                w_pts = np.array([
                    [c * w_step, r * h_step],
                    [(c + 1) * w_step, r * h_step],
                    [(c + 1) * w_step, (r + 1) * h_step],
                    [c * w_step, (r + 1) * h_step]
                ], dtype=np.float32)
                poly = calibration.world_to_px(w_pts).astype(np.int32)

                if d > 0:
                    color = get_grid_color(d)
                    cv2.fillPoly(overlay, [poly], color)
                else:
                    cv2.fillPoly(overlay, [poly], C_EXCL)
                    cv2.line(overlay, tuple(poly[0]), tuple(poly[2]), C_DIM, 1, cv2.LINE_AA)
                    cv2.line(overlay, tuple(poly[1]), tuple(poly[3]), C_DIM, 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        # 2. Draw cell boundaries (white lines) and labels
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                d = cell_densities[r][c]
                w_pts = np.array([
                    [c * w_step, r * h_step],
                    [(c + 1) * w_step, r * h_step],
                    [(c + 1) * w_step, (r + 1) * h_step],
                    [c * w_step, (r + 1) * h_step]
                ], dtype=np.float32)
                poly = calibration.world_to_px(w_pts).astype(np.int32)

                cv2.polylines(frame, [poly], isClosed=True, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

                if d > 0:
                    label = f"{d:.1f}"
                    lw, lh = text_size(label, 0.50)
                    cx = int(np.mean(poly[:, 0]))
                    cy = int(np.mean(poly[:, 1]))
                    lx = cx - lw // 2
                    ly = cy + lh // 2

                    filled_rect(frame, (lx-3, ly-lh-3), (lx+lw+3, ly+4), (0, 0, 0), alpha=0.65)
                    border_rect(frame, (lx-3, ly-lh-3), (lx+lw+3, ly+4), C_CYAN_DIM, 1)
                    put_bold(frame, label, (lx, ly), 0.50, C_TEXT)
    else:
        # Fallback rectangular grid
        cw = w // GRID_COLS
        ch = h // GRID_ROWS

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                d = cell_densities[r][c]
                x1, y1 = c*cw, r*ch
                x2, y2 = x1+cw, y1+ch
                if d > 0:
                    color = get_grid_color(d)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                else:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), C_EXCL, -1)
                    cv2.line(overlay, (x1+10, y1+10), (x2-10, y2-10), C_DIM, 1, cv2.LINE_AA)
                    cv2.line(overlay, (x1+10, y2-10), (x2-10, y1+10), C_DIM, 1, cv2.LINE_AA)

        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)

        for i in range(1, GRID_COLS):
            cv2.line(frame, (i*cw, 0), (i*cw, h), (255,255,255), 1, lineType=cv2.LINE_AA)
        for i in range(1, GRID_ROWS):
            cv2.line(frame, (0, i*ch), (w, i*ch), (255,255,255), 1, lineType=cv2.LINE_AA)

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                d = cell_densities[r][c]
                if d > 0:
                    label = f"{d:.1f}"
                    lw, lh = text_size(label, 0.50)
                    lx = c*cw + 4
                    ly = r*ch + 16
                    filled_rect(frame, (lx-3, ly-lh-3), (lx+lw+3, ly+4), (0, 0, 0), alpha=0.65)
                    border_rect(frame, (lx-3, ly-lh-3), (lx+lw+3, ly+4), C_CYAN_DIM, 1)
                    put_bold(frame, label, (lx, ly), 0.50, C_TEXT)

def draw_congestion_alerts(frame, cell_alerts, frame_idx):
    """Draw warning/critical bounding boxes and labels for congested cells."""
    if not cell_alerts:
        return
        
    for alert in cell_alerts:
        color = C_ORANGE if alert.severity == "WARNING" else C_RED
        
        # Flashing effect based on frame_idx
        t = frame_idx * 0.2
        pulse = int(abs(np.sin(t)) * 100) + 100
        pulse_color = (min(color[0] + pulse, 255), min(color[1] + pulse, 255), min(color[2] + pulse, 255))
        
        if getattr(alert, 'cell_poly', None) is not None:
            # Calibrated: draw perspective quadrilateral border
            pts = np.array(alert.cell_poly, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=pulse_color, thickness=3, lineType=cv2.LINE_AA)
            
            # Find center of cell polygon for text label placement
            moments = cv2.moments(pts)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                cx = int(np.mean(pts[:, 0, 0]))
                cy = int(np.mean(pts[:, 0, 1]))
                
            lbl = alert.severity
            tw, th = text_size(lbl, 0.4, 2)
            filled_rect(frame, (cx - tw//2 - 4, cy - th//2 - 4), (cx + tw//2 + 4, cy + th//2 + 4), (0,0,0), alpha=0.7)
            put_bold(frame, lbl, (cx - tw//2, cy + th//2), 0.4, pulse_color)
        else:
            # Uncalibrated: draw standard rectangular box
            x1, y1, x2, y2 = alert.cell_bounds
            border_rect(frame, (x1, y1), (x2, y2), pulse_color, thickness=3)
            
            lbl = alert.severity
            tw, th = text_size(lbl, 0.4, 2)
            filled_rect(frame, (x1, y1), (x1+tw+8, y1+th+8), (0,0,0), alpha=0.7)
            put_bold(frame, lbl, (x1+4, y1+th+4), 0.4, pulse_color)

def draw_static_exclusions(frame):
    """Visualize ROI boundaries and static obstacle polygons (furniture) on the feed."""
    if not calibration.is_calibrated():
        return

    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # 1. Mask area outside Manual ROI if defined
    if MANUAL_ROI:
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        pts_w = np.array(MANUAL_ROI, dtype=np.float32)
        pts_px = calibration.world_to_px(pts_w).reshape((-1, 1, 2))
        cv2.fillPoly(roi_mask, [pts_px], 255)
        
        # Apply exclusion color to everything outside the ROI
        overlay[roi_mask == 0] = C_EXCL
        cv2.polylines(frame, [pts_px], True, C_CYAN, 2, cv2.LINE_AA)
        put(frame, "MONITORING ZONE", (pts_px[0][0][0], pts_px[0][0][1]-8), 0.35, C_CYAN)

    # 2. Draw Static Obstacles (Furniture, machinery, etc.)
    for obs in STATIC_OBSTACLES:
        pts_w = np.array(obs, dtype=np.float32)
        pts_px = calibration.world_to_px(pts_w).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts_px], C_EXCL)
        cv2.polylines(frame, [pts_px], True, C_DIM, 1, cv2.LINE_AA)
        
        # Label the center of the obstacle
        m = cv2.moments(pts_px)
        if m["m00"] != 0:
            cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
            centered_text(frame, "OBSTACLE", cx, cy, 0.3, C_MUTED)

    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)


def draw_hull(frame, hull_pts):
    if not hull_pts or not isinstance(hull_pts, (list, np.ndarray)):
        return
        
    ov = frame.copy()
    clusters = hull_pts if isinstance(hull_pts, list) else [hull_pts]
    for pts_arr in clusters:
        if pts_arr is None or len(pts_arr) < 3: continue
        pts = pts_arr.reshape(-1, 1, 2)
        cv2.fillPoly(ov, [pts], C_CYAN)
        cv2.polylines(frame, [pts], True, C_CYAN_DIM, 3, cv2.LINE_AA)
        cv2.polylines(frame, [pts], True, C_CYAN,     1, cv2.LINE_AA)
        
    cv2.addWeighted(ov, 0.08, frame, 0.92, 0, frame)

def draw_detections(frame, detections):
    cl = 16   # corner bracket length

    for d in detections:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        cx, cy = d["cx"], d["cy"]

        # Faint box
        cv2.rectangle(frame, (x1, y1), (x2, y2), C_CYAN_DIM, 1)

        # Corner brackets — 2-layer for crispness
        for (px, py, dx, dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame, (px, py), (px+dx*cl, py), C_BORDER2, 3)
            cv2.line(frame, (px, py), (px, py+dy*cl), C_BORDER2, 3)
            cv2.line(frame, (px, py), (px+dx*cl, py), C_TEXT, 1, cv2.LINE_AA)
            cv2.line(frame, (px, py), (px, py+dy*cl), C_TEXT, 1, cv2.LINE_AA)

        # Center dot with glow
        glow_circle(frame, (cx, cy), 3, C_CYAN, alpha=0.25, layers=2)
        cv2.circle(frame, (cx, cy), 2, C_CYAN, -1, cv2.LINE_AA)

        # ID tag — pill shape
        if d["pid"] >= 0:
            tag = f"#{d['pid']:03d}"
            tw, th = text_size(tag, 0.36)
            tx, ty = x1, y1 - th - 10
            # shadow
            filled_rect(frame, (tx-1, ty-3), (tx+tw+11, ty+th+3), (0,0,0), alpha=0.6)
            # bg
            filled_rect(frame, (tx,   ty-2), (tx+tw+10, ty+th+2), C_CARD)
            border_rect(frame, (tx,   ty-2), (tx+tw+10, ty+th+2), C_CYAN_DIM)
            put(frame, tag, (tx+5, ty+th-1), 0.36, C_CYAN, 1)


def draw_corner_brackets(frame, h, w):
    bl = 28
    th = 3
    for (px2, py2, dx, dy) in [(0,0,1,1),(w,0,-1,1),(0,h,1,-1),(w,h,-1,-1)]:
        # Outer glow
        cv2.line(frame, (px2,py2), (px2+dx*bl, py2), C_CYAN_DIM, th+2, cv2.LINE_AA)
        cv2.line(frame, (px2,py2), (px2, py2+dy*bl), C_CYAN_DIM, th+2, cv2.LINE_AA)
        # Crisp line
        cv2.line(frame, (px2,py2), (px2+dx*bl, py2), C_CYAN, th, cv2.LINE_AA)
        cv2.line(frame, (px2,py2), (px2, py2+dy*bl), C_CYAN, th, cv2.LINE_AA)
        # Inner dot
        cv2.circle(frame, (px2, py2), 3, C_CYAN, -1, cv2.LINE_AA)


def draw_hull_area_label(frame, hull_area_m2, h, w):
    """Display calculated crowd hull area (automatically computed from detections) - LARGER & CLEARER."""
    # Area is automatically calculated from the convex hull of all detected people
    label = f"CROWD AREA  {hull_area_m2:.1f} m²"
    tw, th = text_size(label, 0.55)  # Increased from 0.40
    cx = w // 2
    pad_x, pad_y = 18, 8  # Increased padding

    bx1 = cx - tw//2 - pad_x
    by1 = h - th - pad_y*2 - 8
    bx2 = cx + tw//2 + pad_x
    by2 = h - 8

    filled_rect(frame, (bx1, by1), (bx2, by2), C_CARD, alpha=0.95)  # More opaque
    border_rect(frame, (bx1, by1), (bx2, by2), C_CYAN, thickness=3)  # Thicker border

    # Accent left stripe (thicker)
    cv2.line(frame, (bx1, by1+1), (bx1, by2-1), C_CYAN, 4)

    centered_text_bold(frame, label, cx, (by1+by2)//2, 0.55, C_CYAN)


# ─────────────────────────────────────────────────────────────────────────────
# Top bar
# ─────────────────────────────────────────────────────────────────────────────

def draw_top_bar(frame, w, frame_idx):
    BAR_H = 46

    # Background + subtle gradient feel (manual 2-strip)
    filled_rect(frame, (0, 0),       (w, BAR_H//2), C_PANEL2)
    filled_rect(frame, (0, BAR_H//2),(w, BAR_H),    C_PANEL)
    hline(frame, 0, w, BAR_H, C_BORDER, 1)
    hline(frame, 0, w, BAR_H-1, C_BORDER2, 1)  # double-border crispness

    # REC indicator — animated pulsing ring
    rec_cx, rec_cy = 18, BAR_H // 2
    if frame_idx % 60 < 45:
        pulse_r = 5 + (frame_idx % 30) // 15  # subtle size pulse
        glow_circle(frame, (rec_cx, rec_cy), pulse_r, C_RED, alpha=0.2, layers=2)
        cv2.circle(frame, (rec_cx, rec_cy), 5, C_RED, -1, cv2.LINE_AA)
        cv2.circle(frame, (rec_cx, rec_cy), 5, (180, 50, 50), 1, cv2.LINE_AA)
    put(frame, "REC", (rec_cx + 9, rec_cy + 4), 0.30, C_RED, 1)

    # System title
    title = "AI CROWD DENSITY MONITOR"
    put_bold(frame, title, (48, BAR_H//2 + 5), 0.42, C_TEXT)

    # Separator
    vline(frame, 310, 8, BAR_H-8, C_BORDER, 1)

    # Place badge
    place = get_place().upper()
    place_lbl = f"ZONE: {place}"
    tw_p, _ = text_size(place_lbl, 0.38)
    px_pl = 322
    filled_rect(frame, (px_pl-4, BAR_H//2-10),
                (px_pl+tw_p+10, BAR_H//2+10), C_CARD)
    border_rect(frame, (px_pl-4, BAR_H//2-10),
                (px_pl+tw_p+10, BAR_H//2+10), C_CYAN_DIM)
    put(frame, place_lbl, (px_pl+2, BAR_H//2+5), 0.38, C_CYAN, 1)

    # Timestamp
    ts = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    tw_ts, _ = text_size(ts, 0.38)
    ts_x = w - tw_ts - 14
    put(frame, ts, (ts_x, BAR_H//2 + 5), 0.38, C_MUTED, 1)

    # CAM-01 badge (left of timestamp)
    cam = "CAM-01"
    tw_c, _ = text_size(cam, 0.36)
    cx_c = ts_x - tw_c - 26
    filled_rect(frame, (cx_c-4, BAR_H//2-9), (cx_c+tw_c+8, BAR_H//2+9), C_CARD)
    border_rect(frame, (cx_c-4, BAR_H//2-9), (cx_c+tw_c+8, BAR_H//2+9), C_BORDER)
    put(frame, cam, (cx_c, BAR_H//2+5), 0.36, C_BLUE, 1)

    return BAR_H


# ─────────────────────────────────────────────────────────────────────────────
# Right-side HUD panel
# ─────────────────────────────────────────────────────────────────────────────

def draw_side_panel(frame, h, w, bar_h,
                    stable_count, smooth_density, phys_density, room_density, hull_area_m2,
                    overlap_ratio, fps, risk, risk_color,
                    density_history, zone_counts, alert_active, frame_idx,
                    hull_type="None", alpha_value=0.0):

    PW = 300  # Slightly wider for better readability
    px = w - PW
    py = bar_h

    # Panel background — two-tone
    filled_rect(frame, (px, py), (w, h), C_PANEL)
    filled_rect(frame, (px, py), (px+1, h), C_BORDER)  # crisp left edge

    # Subtle inner border
    vline(frame, px,   py, h, C_BORDER,  1)
    vline(frame, px+1, py, h, C_BORDER2, 1)

    y = py + 14

    # ── CROWD ANALYTICS ─────────────────────────────────────────────────
    section_header(frame, "CROWD ANALYTICS", px, y, PW)
    y += 18

    CW = (PW - 30) // 2
    CH = 72  # Increased from 56

    def stat_card(x, y_, val_str, lbl, val_color=C_TEXT, unit=""):
        # card bg + border (darker, more visible)
        filled_rect(frame, (x, y_), (x+CW, y_+CH), C_CARD2)
        border_rect(frame, (x, y_), (x+CW, y_+CH), val_color, thickness=2)
        # left accent stripe (thicker)
        cv2.line(frame, (x, y_+3), (x, y_+CH-3), val_color, 3)
        
        # value - LARGER FONT
        tw_v, th_v = text_size(val_str, 0.75, 2)
        # background for value for better contrast
        filled_rect(frame, (x + CW//2 - tw_v//2 - 4, y_ + 22 - th_v - 2),
                    (x + CW//2 + tw_v//2 + 4, y_ + 22 + 8), 
                    (0, 0, 0), alpha=0.4)
        put_bold(frame, val_str,
                 (x + CW//2 - tw_v//2, y_ + 30),
                 0.75, val_color)
        
        # unit (if any) - LARGER
        if unit:
            tw_u, _ = text_size(unit, 0.32)
            put(frame, unit, (x + CW//2 + tw_v//2 - tw_u + 4, y_ + 24),
                0.32, C_CYAN)
        
        # label - MUCH LARGER AND BRIGHTER
        tw_lbl, th_lbl = text_size(lbl, 0.38)
        # background for label
        filled_rect(frame, (x + CW//2 - tw_lbl//2 - 3, y_ + CH - 16 - th_lbl),
                    (x + CW//2 + tw_lbl//2 + 3, y_ + CH - 4), 
                    val_color, alpha=0.15)
        centered_text(frame, lbl, x + CW//2, y_ + CH - 9, 0.38, C_TEXT, 1)

    col1 = px + 10
    col2 = px + 10 + CW + 8

    stat_card(col1, y, str(stable_count),         "COUNT",     C_TEXT,   "pax")
    stat_card(col2, y, f"{hull_area_m2:.1f}",     "FOOTPRINT", C_BLUE,   "m²")
    y += CH + 10

    # ── SPACE INTENSITY ─────────────────────────────────────────────────
    section_header(frame, "SPACE INTENSITY", px, y, PW)
    y += 18
    
    stat_card(col1, y, f"{phys_density:.2f}",     "CROWD D.",  C_CYAN,   "p/m²")
    stat_card(col2, y, f"{room_density:.2f}",     "ROOM D.",   C_TEAL,   "p/m²")
    y += CH + 10
    
    stat_card(col1, y, f"{smooth_density:.0f}",   "ROOM LOAD", C_PURPLE, "%")
    stat_card(col2, y, f"{fps:.1f}",              "PERF",      C_GREEN,  "fps")
    y += CH + 12

    # ── ALGORITHM METADATA ──────────────────────────────────────────────
    # Small indicator for Alpha Shape settings
    put(frame, f"METHOD: {hull_type}", (px + 12, y), 0.32, C_MUTED)
    put(frame, f"ALPHA:  {alpha_value:.2f}", (px + 12, y + 14), 0.32, C_MUTED)
    y += 24

    hline(frame, px+6, w-6, y, C_BORDER2, 1)
    y += 12

    # ── RISK LEVEL ──────────────────────────────────────────────────────
    section_header(frame, "RISK LEVEL", px, y, PW)
    y += 18

    # Risk label — large, right-aligned
    rs = 0.56
    tw_r, th_r = text_size(risk, rs, 2)
    put_bold(frame, risk, (w - tw_r - 12, y + th_r + 2), rs, risk_color)

    # LOS bar background
    BAR_W = PW - 20
    bar_y = y + th_r + 14

    # Track
    filled_rect(frame, (px+10, bar_y), (px+10+BAR_W, bar_y+8), C_CARD)
    border_rect(frame, (px+10, bar_y), (px+10+BAR_W, bar_y+8), C_BORDER)

    # Fill with gradient: green → yellow → red
    fill_w = int(BAR_W * LOS_WIDTHS.get(risk, 0.5))
    if fill_w > 0:
        draw_gradient_bar(frame,
                          px+10, bar_y+1, px+10+fill_w, bar_y+7,
                          C_GREEN, risk_color)

    # Tick marks + labels
    bar_y2 = bar_y + 10
    for label, frac in [("0.5", 0.14), ("0.8", 0.28), ("1.26", 0.55), ("2.0", 0.85)]:
        tx = px + 10 + int(BAR_W * frac)
        cv2.line(frame, (tx, bar_y-1), (tx, bar_y+9), C_BORDER2, 1)
        tw_t, _ = text_size(label, 0.27)
        put(frame, label, (tx - tw_t//2, bar_y2 + 8), 0.27, C_DIM)

    y = bar_y2 + 18
    hline(frame, px+6, w-6, y, C_BORDER2, 1)
    y += 12

    # ── DENSITY TREND ───────────────────────────────────────────────────
    section_header(frame, "DENSITY TREND", px, y, PW)
    y += 18

    GH = 44   # graph height
    GW = PW - 20

    # Graph background
    filled_rect(frame, (px+10, y), (px+10+GW, y+GH), C_CARD)
    border_rect(frame, (px+10, y), (px+10+GW, y+GH), C_BORDER)

    if density_history:
        mx = max(max(density_history), 0.1)
        bw = max(1, GW // max(len(density_history), 1))

        for i, val in enumerate(density_history):
            bh_val = int((val / mx) * (GH - 4))
            bx2 = px + 10 + i * bw
            bc = get_grid_color(val) if val > 0 else C_CARD2
            # bar fill
            filled_rect(frame, (bx2+1, y + GH - bh_val - 1),
                        (bx2+bw-1, y + GH - 1), bc)
            # bright top edge
            if bh_val > 0:
                cv2.line(frame, (bx2+1, y+GH-bh_val-1),
                         (bx2+bw-1, y+GH-bh_val-1), C_TEXT, 1)

        # Dashed midline
        mid_y = y + GH // 2
        for dash_x in range(px+12, px+10+GW-2, 5):
            cv2.line(frame, (dash_x, mid_y), (dash_x+2, mid_y), C_BORDER2, 1)

        # Y-axis labels
        put(frame, f"{mx:.1f}", (px+12, y+10),      0.26, C_DIM)
        put(frame, "0.0",       (px+12, y+GH-2),    0.26, C_DIM)

    y += GH + 14
    hline(frame, px+6, w-6, y, C_BORDER2, 1)
    y += 12

    # ── ZONE MAP ────────────────────────────────────────────────────────
    section_header(frame, "ZONE MAP", px, y, PW)
    y += 18

    cell_w = (PW - 20) // GRID_COLS
    cell_h = 24  # Increased from 22

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cnt = zone_counts[r][c]
            zx1 = px + 10 + c * cell_w
            zy1 = y + r * cell_h
            if cnt:
                color = get_grid_color(cnt / 2.0)
                # cell bg
                filled_rect(frame, (zx1+1, zy1+1),
                            (zx1+cell_w-2, zy1+cell_h-2), color, 0.65)  # More opaque
                border_rect(frame, (zx1+1, zy1+1),
                            (zx1+cell_w-2, zy1+cell_h-2), C_BORDER2, 1)
                # LARGER, BOLD text for zone count
                centered_text_bold(frame, str(cnt),
                              zx1 + cell_w//2, zy1 + cell_h//2,
                              0.42, (255, 255, 255))
            else:
                # Draw "Excluded" state in Zone Map
                filled_rect(frame, (zx1+1, zy1+1), (zx1+cell_w-2, zy1+cell_h-2), C_EXCL, 0.7)
                cv2.line(frame, (zx1+5, zy1+5), (zx1+cell_w-5, zy1+cell_h-5), C_DIM, 1, cv2.LINE_AA)

    y += GRID_ROWS * cell_h + 10

    hline(frame, px+6, w-6, y, C_BORDER2, 1)
    y += 10

    # FPS pill (bottom of panel)
    fps_txt = f"FPS {fps:.1f}"
    tw_f, th_f = text_size(fps_txt, 0.34)
    fx = px + 10
    filled_rect(frame, (fx-2, y),    (fx+tw_f+10, y+th_f+8), C_CARD)
    border_rect(frame, (fx-2, y),    (fx+tw_f+10, y+th_f+8), C_GREEN)
    cv2.line(frame, (fx-2, y+1), (fx-2, y+th_f+7), C_GREEN, 2)
    put(frame, fps_txt, (fx+4, y+th_f+3), 0.34, C_GREEN, 1)

    # ── ALERT FLASH OVERLAY ─────────────────────────────────────────────
    if alert_active:
        t = frame_idx * 0.15
        pulse = int(abs(np.sin(t)) * 90)
        filled_rect(frame, (px, py), (w, h), (0, 0, max(pulse, 15)), alpha=0.35)

        # Animated "!! ALERT !!" text
        alert_txt = "!! ALERT ACTIVE !!"
        tw_a, th_a = text_size(alert_txt, 0.52, 2)
        ay = h - 40
        ax = px + PW//2 - tw_a//2
        # shadow
        put(frame, alert_txt, (ax+1, ay+1), 0.52, (0, 0, 80), 2)
        put(frame, alert_txt, (ax,   ay),   0.52, C_RED,       2)


# ─────────────────────────────────────────────────────────────────────────────
# Bottom status bar
# ─────────────────────────────────────────────────────────────────────────────

def draw_bottom_bar(frame, fps, hull_area_m2, overlap_ratio, h, w, PW=220):
    BAR_H = 32
    y = h - BAR_H
    filled_rect(frame, (0, y), (w - PW, h), C_PANEL)
    hline(frame, 0, w - PW, y, C_BORDER, 1)
    hline(frame, 0, w - PW, y+1, C_BORDER2, 1)

    def pill(text, x, fg, bg):
        tw, th = text_size(text, 0.36)
        x1p, x2p = x, x + tw + 18
        y1p, y2p = y + 6, y + BAR_H - 6
        filled_rect(frame, (x1p, y1p), (x2p, y2p), bg)
        border_rect(frame, (x1p, y1p), (x2p, y2p), fg)
        cv2.line(frame, (x1p, y1p+2), (x1p, y2p-2), fg, 2)
        put(frame, text, (x1p + 9, y + BAR_H//2 + 5), 0.36, fg, 1)
        return x2p + 8

    x = 10
    x = pill(f"FPS {fps:.1f}",                     x, C_GREEN,  (10, 28, 14))
    x = pill(f"Hull {hull_area_m2:.1f} m²",        x, C_BLUE,   (10, 22, 35))
    x = pill(f"Occlusion {overlap_ratio*100:.0f}%", x, C_PURPLE, (22, 14, 38))

    # Legend — right side of bottom bar (excluding panel)
    legend = [
        ("EXCL",  C_EXCL),
        ("EMPTY", C_GREEN),
        ("LOW",   C_TEAL),
        ("MOD",   C_YELLOW),
        ("HIGH",  C_ORANGE),
        ("CRIT",  C_RED),
    ]
    rx = w - PW - 8
    for lbl, col in reversed(legend):
        tw, _ = text_size(lbl, 0.30)
        rx -= tw + 22
        cv2.rectangle(frame, (rx, y+10), (rx+10, y+22), col, -1)
        cv2.rectangle(frame, (rx, y+10), (rx+10, y+22), C_BORDER2, 1)
        put(frame, lbl, (rx + 13, y + 21), 0.30, C_MUTED)

    # Center quit hint
    hint = "[Q] Quit"
    tw_q, _ = text_size(hint, 0.34)
    fw = w - PW
    put(frame, hint, (fw//2 - tw_q//2, y + BAR_H//2 + 5), 0.34, C_DIM)


# ─────────────────────────────────────────────────────────────────────────────
# Cinematic / post-processing effects
# ─────────────────────────────────────────────────────────────────────────────

def add_vignette(img, strength=0.50):
    h, w = img.shape[:2]
    X = cv2.getGaussianKernel(w, int(w * 0.52))
    Y = cv2.getGaussianKernel(h, int(h * 0.52))
    mask = Y @ X.T
    mask = mask / mask.max()
    mask = mask * (1 - strength) + strength
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c] * mask, 0, 255).astype(np.uint8)


def add_noise(img, strength=2):
    noise = np.random.randint(-strength, strength+1,
                              img.shape, dtype=np.int16)
    img[:] = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def add_scanlines(img, gap=3, alpha=0.08):
    """Subtle CRT scanline effect for visual depth."""
    h, w = img.shape[:2]
    ov = img.copy()
    for row in range(0, h, gap):
        cv2.line(ov, (0, row), (w, row), (0, 0, 0), 1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)


def add_chromatic_aberration(img, shift=1):
    """Very subtle channel shift for cinematic look."""
    if shift <= 0:
        return
    h, w = img.shape[:2]
    M_pos = np.float32([[1, 0,  shift], [0, 1, 0]])
    M_neg = np.float32([[1, 0, -shift], [0, 1, 0]])
    img[:, :, 2] = cv2.warpAffine(img[:, :, 2], M_pos, (w, h))   # R
    img[:, :, 0] = cv2.warpAffine(img[:, :, 0], M_neg, (w, h))   # B


# ─────────────────────────────────────────────────────────────────────────────
# Master render call
# ─────────────────────────────────────────────────────────────────────────────

def render_frame(frame, detections, cell_densities, density_history, 
                 stable_count, smooth_density, phys_density, room_density, overlap_ratio, fps,
                 risk, risk_color, frame_idx, alert_active,
                 hull_pts=None, hull_area_m2=0.0, kde_map=None,
                 zone_counts=None,
                 hull_type="None", alpha_value=0.0, cell_alerts=None):

    if zone_counts is None:
        zone_counts = [[0] * GRID_COLS for _ in range(GRID_ROWS)]

    h, w = frame.shape[:2]
    PW      = 300          # side panel width — must match draw_side_panel's PW
    feed_w  = w - PW

    # ── Feed area ─────────────────────────────────────────────────────
    feed = frame[:, :feed_w].copy()
    feed_h, feed_w_px = feed.shape[:2]

    draw_grid_heatmap(feed, cell_densities)
    draw_static_exclusions(feed)
    draw_hull(feed, hull_pts)

    # 1.5 Draw KDE Heatmap Overlay
    if kde_map is not None and kde_map.max() > 0:
        calibrated = calibration.is_calibrated()
        H_inv = calibration.get_homography_inv() if calibrated else None
        
        if calibrated and H_inv is not None:
            res = kde_map.shape[0]  # resolution (KDE_RESOLUTION)
            S = np.array([
                [WORLD_GRID_W / res, 0, 0],
                [0, WORLD_GRID_H / res, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            M = H_inv.dot(S)
            kde_resized = cv2.warpPerspective(kde_map, M, (feed_w_px, feed_h))
        else:
            kde_resized = cv2.resize(kde_map, (feed_w_px, feed_h))

        kde_color = cv2.applyColorMap((kde_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        mask = (kde_resized > 0.05).astype(np.uint8) * 255
        kde_overlay = cv2.bitwise_and(kde_color, kde_color, mask=mask)
        feed = cv2.addWeighted(feed, 1.0, kde_overlay, 0.45, 0)

    draw_detections(feed, detections)
    draw_congestion_alerts(feed, cell_alerts, frame_idx)
    draw_corner_brackets(feed, feed_h, feed_w_px)
    draw_hull_area_label(feed, hull_area_m2, feed_h, feed_w_px)

    add_vignette(feed, 0.45)
    add_scanlines(feed, gap=4, alpha=0.06)
    add_chromatic_aberration(feed, shift=1)
    add_noise(feed, 2)

    frame[:, :feed_w_px] = feed

    # ── Overlays (full-width) ─────────────────────────────────────────
    bar_h = draw_top_bar(frame, w, frame_idx)

    draw_side_panel(frame, h, w, bar_h,
                    stable_count, smooth_density, phys_density, room_density, hull_area_m2,
                    overlap_ratio, fps, risk, risk_color,
                    density_history, zone_counts, alert_active, frame_idx,
                    hull_type=hull_type, alpha_value=alpha_value)

    draw_bottom_bar(frame, fps, hull_area_m2, overlap_ratio, h, w, PW)

    return cv2.resize(frame, (DISPLAY_W, DISPLAY_H),
                      interpolation=cv2.INTER_LINEAR)