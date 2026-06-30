# main.py — Crowd Monitoring System with Live Calibration Support
#
# Key Features:
# • Ground-plane homography calibration with live preview
# • Real-time crowd density with perspective correction
# • Recalibration support (press 'C' during monitoring)
# • Comprehensive logging and alerting
# • Context-aware risk assessment

import cv2
import time
import os
import numpy as np
import threading

from config import (
    CAMERA_INDEX,
    CAPTURE_W,
    CAPTURE_H,
    DISPLAY_W,
    DISPLAY_H,
    GRID_ROWS,
    GRID_COLS
)

from detector   import load_model, run_detection
from density    import DensityTracker
from flow       import FlowTracker, draw_flow
from cnn_model  import DensityCNN
from alert      import AlertManager
from logger     import init_log, log_frame
from ui         import render_frame, select_place
from context_risk import get_context_risk, get_place
import calibration

from ground_segmentor import GroundSegmentor
from temporal_filter import TemporalFilter
from congestion import CongestionDetector


# ─────────────────────────────────────────────────────────────────────────────
# Calibration management
# ─────────────────────────────────────────────────────────────────────────────

def startup_calibration_check():
    """
    Always ask the user to click 4 floor corners on every startup.
    If confirmed  → saves homography.npy and uses perspective-correct density.
    If S / ESC    → loads any previously saved calibration, or falls back to
                    the uniform CELL_AREA_M2 constant.
    """
    print("\n" + "="*70)
    print("CALIBRATION  (runs every startup)")
    print("="*70)
    print()
    print("  Click the 4 corners of the floor area in the video frame.")
    print("  Press ENTER to confirm  |  S or ESC to skip / use last saved.")
    print()

    try:
        # Capture a representative frame from the video / camera
        cap = cv2.VideoCapture(calibration.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  calibration.CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, calibration.CAPTURE_H)
        grabbed = None
        for _ in range(30):
            ret, f = cap.read()
            if ret:
                grabbed = f
        cap.release()

        # Open manual 4-point UI every time
        manual_pts = calibration._manual_point_selection(grabbed)

        if manual_pts is not None:
            # User confirmed 4 points — compute and save new homography
            H = calibration._compute_homography_from_pattern(
                manual_pts, (1, 1), "tile_grid"
            )
            if H is not None:
                calibration.load_homography()
                print("✓ Calibration saved! Using perspective-correct density.")
                return True

    except Exception as e:
        print(f"  Calibration error: {e}")

    # User skipped — try to load any previously saved homography
    H = calibration.load_homography()
    if H is not None:
        print("✓ Loaded previous calibration from homography.npy.")
        return True

    print("⚠  No calibration — using uniform CELL_AREA_M2 fallback.")
    print("  Tip: Press C during monitoring to calibrate at any time.")
    return False


def runtime_recalibration_menu():
    """
    Show menu for recalibration/validation during monitoring.
    """
    print("\n" + "="*70)
    print("CALIBRATION MENU")
    print("="*70)
    
    if calibration.is_calibrated():
        print("Current status: ✓ Calibrated")
        print("Options:")
        print("  [V] Validate current calibration")
        print("  [R] Recalibrate")
        print("  [Q] Return to monitoring")
        choice = input("\nEnter choice (V/R/Q): ").strip().upper()
        
        if choice == "V":
            print("\nValidating calibration...")
            try:
                cap = cv2.VideoCapture(CAMERA_INDEX)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    if calibration.validate_calibration(test_frame=frame):
                        print("✓ Calibration validation: PASSED")
                    else:
                        print("✗ Calibration validation: REJECTED")
                else:
                    print("✗ Cannot read from camera.")
            except Exception as e:
                print(f"✗ Validation error: {e}")
                
        elif choice == "R":
            print("\nStarting recalibration...")
            if calibration.recalibrate_interactive():
                print("✓ Recalibration successful!")
            else:
                print("✗ Recalibration failed or cancelled.")
    else:
        print("Current status: ✗ Not calibrated (using fallback)")
        print("Options:")
        print("  [C] Calibrate now")
        print("  [Q] Return to monitoring")
        choice = input("\nEnter choice (C/Q): ").strip().upper()
        
        if choice == "C":
            print("\nStarting calibration...")
            try:
                if calibration.run_live_calibration_ui():
                    print("✓ Calibration successful!")
                else:
                    print("✗ Calibration failed.")
            except Exception as e:
                print(f"✗ Calibration error: {e}")
    
    print("\nResuming monitoring...\n")
    time.sleep(1)


# ─────────────────────────────────────────────────────────────────────────────
# Threaded AI Engine (Maximum FPS)
# ─────────────────────────────────────────────────────────────────────────────

class DetectionWorker(threading.Thread):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.frame = None
        self.h = 0
        self.w = 0
        self.running = True
        
        # Initial empty state
        self.detections = []
        self.count = 0
        self.overlap = 0
        self.zone_counts = [[0]*GRID_COLS for _ in range(GRID_ROWS)]
        
        self.lock = threading.Lock()
        
    def update_frame(self, frame, h, w):
        with self.lock:
            if self.frame is None:  # Only accept new frame if previous is finished
                self.frame = frame.copy()
                self.h = h
                self.w = w
            
    def get_results(self):
        with self.lock:
            return self.detections, self.count, self.overlap, self.zone_counts
            
    def run(self):
        while self.running:
            f, h, w = None, 0, 0
            with self.lock:
                if self.frame is not None:
                    f = self.frame
                    h = self.h
                    w = self.w
                    
            if f is not None:
                # Run heavy YOLO + zoomed tiling pass
                dets, _, cnt, ov, zc = run_detection(self.model, f, h, w)
                
                with self.lock:
                    self.detections = dets
                    self.count = cnt
                    self.overlap = ov
                    self.zone_counts = zc
                    self.frame = None # Mark as ready for next frame
            else:
                time.sleep(0.005) # Yield CPU
                
    def stop(self):
        self.running = False


# ─────────────────────────────────────────────────────────────────────────────
# Risk colour
# ─────────────────────────────────────────────────────────────────────────────

def get_risk_color(risk):
    return {
        "VERY LOW": (80,  220, 100),
        "LOW":      (0,   220, 255),
        "MODERATE": (0,   200, 255),
        "HIGH":     (0,   140, 255),
        "CRITICAL": (60,  60,  255),
    }.get(risk, (0, 0, 255))


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():

    # 1. Calibration
    startup_calibration_check()

    # 2. Select monitoring zone
    select_place()
    print("Selected Place:", get_place())

    use_auto_seg = False # Auto-selected no to avoid blocking console input
    segmentor = GroundSegmentor() if use_auto_seg else None

    # 3. Init log file
    init_log()

    # 4. Load YOLO
    model = load_model()

    # 5. Camera
    cam_source = CAMERA_INDEX
    if isinstance(cam_source, str) and ("youtube.com" in cam_source or "youtu.be" in cam_source):
        print(f"\n[STREAM] Resolving YouTube URL: {cam_source}...")
        import yt_dlp
        ydl_opts = {'format': 'best[ext=mp4]/best'}
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(cam_source, download=False)
                cam_source = info['url']
                print("[STREAM] Resolved stream successfully.")
        except Exception as e:
            print(f"[STREAM] Error resolving URL: {e}")
            
    cap = cv2.VideoCapture(cam_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)

    cv2.namedWindow("Crowd Density Monitoring", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Crowd Density Monitoring", DISPLAY_W, DISPLAY_H)

    # 6. Trackers
    density_tracker = DensityTracker()
    flow_tracker    = FlowTracker()
    cnn_model       = DensityCNN() if os.path.exists(os.path.join("models", "crowd_cnn.pt")) else None
    alert_manager   = AlertManager()
    
    # 7. Robustness Filters
    temporal_filter = TemporalFilter()
    congestion_detector = CongestionDetector(
        roi_polygon=None,  # We'll use the default full frame initially
        frame_shape=(CAPTURE_H, CAPTURE_W),
        rows=GRID_ROWS, cols=GRID_COLS
    )

    # 8. Threaded AI Worker
    # We offload the heavy AI to a background thread.
    worker = DetectionWorker(model)
    worker.start()

    frame_idx = 0
    prev      = time.time()
    
    print("\n" + "="*70)
    print("MONITORING STARTED")
    print("="*70)
    print("Controls:")
    print("  [Q] Quit")
    print("  [C] Calibration menu (validate/recalibrate)")
    print("="*70 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            # Video ended, loop it
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        frame_idx += 1
        h, w = frame.shape[:2]

        # ── Auto Segmentation (Run once) ──────────────────────────────────
        if frame_idx == 1 and segmentor and segmentor.is_available:
            print("\n[AI] Running ground segmentation on first frame...")
            mask = segmentor.segment(frame)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                pts_px = largest_contour.reshape(-1, 2)
                if calibration.is_calibrated():
                    import config
                    pts_w = calibration.px_to_world(pts_px.tolist())
                    config.MANUAL_ROI = pts_w.tolist()
                    print(f"[AI] Auto-generated monitoring zone with {len(pts_w)} points.")

        # ── Detection (Non-Blocking Threaded) ─────────────────────────────
        # Send current frame to background thread
        worker.update_frame(frame, h, w)
        
        # Instantly grab latest available results (might be a few ms old)
        detections, count, overlap, zone_counts = worker.get_results()
        
        # ── Temporal Smoothing ────────────────────────────────────────────
        zone_counts_np = np.array(zone_counts, dtype=np.float32)
        smoothed_counts = temporal_filter.update(zone_counts_np)
        
        # ── Cell-Level Congestion Alerts ──────────────────────────────────
        _, cell_alerts = congestion_detector.analyze(detections)

        # ── CNN Density Estimation ────────────────────────────────────────
        cnn_count, cnn_map = 0.0, None
        if cnn_model:
            cnn_count, cnn_map = cnn_model.predict(frame)

        # ── Density (real-world) ───────────────────────────────────────────
        (
            stable_count,
            density,
            phys_density,
            room_density,
            cells,
            hull,
            area,
            kde_map,
            hull_type,
            alpha_value
        ) = density_tracker.update(count, zone_counts, detections, w, h, cnn_count=cnn_count)

        # ── Flow (real-world velocities) ───────────────────────────────────
        vectors, cell_flow = flow_tracker.update(detections)

        # ── Context-aware risk ─────────────────────────────────────────────
        risk  = get_context_risk(stable_count)
        color = get_risk_color(risk)

        # ── FPS ───────────────────────────────────────────────────────────
        now = time.time()
        fps = 1 / max(now - prev, 0.001)
        prev = now

        # ── Alert ─────────────────────────────────────────────────────────
        annotated = frame.copy()    # alert recording uses the annotated frame
        alert_active = alert_manager.update(risk, frame, frame_idx, annotated)

        # ── Logging ───────────────────────────────────────────────────────
        log_frame(frame_idx, stable_count, density, area, risk, overlap, fps)

        # ── Draw flow on frame before render ──────────────────────────────
        draw_flow(frame, vectors, cell_flow, h, w)

        # ── UI render ─────────────────────────────────────────────────────
        output = render_frame(
            frame=frame.copy(),
            detections=detections,
            cell_densities=cells,
            density_history=density_tracker.history(),
            stable_count=stable_count,
            smooth_density=density,
            overlap_ratio=overlap,
            fps=fps,
            risk=risk,
            risk_color=color,
            frame_idx=frame_idx,
            alert_active=alert_active,
            hull_pts=hull,
            hull_area_m2=area,
            zone_counts=smoothed_counts.tolist(), # Use smoothed counts
            kde_map=kde_map,
            phys_density=phys_density,
            room_density=room_density,
            hull_type=hull_type,
            alpha_value=alpha_value,
            cell_alerts=cell_alerts # Pass congestion alerts
        )

        cv2.imshow("Crowd Density Monitoring", output)

        # ── Keyboard input ────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            print("\n✓ Shutting down...")
            break
        elif key == ord("c") or key == ord("C"):
            # Pause monitoring for calibration menu
            print("\n[MONITORING] Paused for calibration menu...")
            runtime_recalibration_menu()

    # ── Cleanup ───────────────────────────────────────────────────────────
    worker.stop()
    worker.join()
    alert_manager.release()
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Monitoring stopped.\n")


if __name__ == "__main__":
    main()
