import cv2
import time
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import load_model, run_detection
from density import DensityTracker
from ui import render_frame
from cnn_model import DensityCNN
from context_risk import get_context_risk
import calibration
from config import MODEL_PATH, DISPLAY_W, DISPLAY_H

def get_risk_color(risk):
    return {
        "VERY LOW": (80,  220, 100),
        "LOW":      (0,   220, 255),
        "MODERATE": (0,   200, 255),
        "HIGH":     (0,   140, 255),
        "CRITICAL": (60,  60,  255),
    }.get(risk, (0, 0, 255))

def main():
    # Load the model
    model = load_model()
    
    video_path = input("Enter the path to your video or GIF (e.g., giphy.gif): ").strip()
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Tracking states
    density_tracker = DensityTracker()
    cnn_model = DensityCNN()
    
    cv2.namedWindow("Video Inference Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video Inference Test", DISPLAY_W, DISPLAY_H)

    frame_idx = 0
    prev = time.time()

    print(f"Processing: {video_path}")
    print("Press 'q' to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_idx += 1
        h, w = frame.shape[:2]

        # 1. Detection
        detections, _, count, overlap, zone_counts = run_detection(model, frame, h, w)

        # 1.5 CNN Density
        cnn_count, cnn_map = cnn_model.predict(frame)

        # 2. Density Calculation (DBSCAN + Alpha Shapes)
        (stable_count, density, phys_density, cells, hull, area, hull_type, alpha_value) = \
            density_tracker.update(count, zone_counts, detections, w, h, cnn_count=cnn_count)

        # 3. Risk Assessment
        risk = get_context_risk(stable_count)
        color = get_risk_color(risk)

        # 4. UI Rendering
        output = render_frame(
            frame.copy(), detections, cells, density_tracker.history(),
            stable_count, density, phys_density, overlap, 0.0, risk, color,
            frame_idx, False, hull, area, zone_counts=zone_counts,
            hull_type=hull_type, alpha_value=alpha_value
        )

        cv2.imshow("Video Inference Test", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Inference complete.")

if __name__ == "__main__":
    main()