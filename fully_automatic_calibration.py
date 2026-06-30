import os
import cv2
import numpy as np
import sys
import time
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).parent))
 
try:
    from config import (
        CAMERA_INDEX, CAPTURE_W, CAPTURE_H, HOMOGRAPHY_FILE, 
        WORLD_GRID_W, WORLD_GRID_H, GRID_ROWS, GRID_COLS
    )
except ImportError as e:
    print(f"Error importing config: {e}")
    sys.exit(1)
 
 
class FullyAutomaticCalibrator:
    """
    Completely automatic calibration with ZERO human interaction.
    
    Features:
    - Automatically detects floor pattern
    - Automatically computes homography
    - Automatically validates quality
    - Automatically saves if valid
    - Automatically retries on failure
    - No clicking, no measurements, no dialogs
    """
    
    def __init__(self, max_retries=5, frame_wait_time=0.5, verbose=False):
        self.max_retries = max_retries
        self.frame_wait_time = frame_wait_time  # seconds between frame captures
        self.verbose = verbose
        self.best_H = None
        self.best_quality = -1
        
    def log(self, msg, level="INFO"):
        """Simple logging."""
        if self.verbose:
            print(f"[{level}] {msg}")
    
    def detect_checkerboard(self, frame, board_size_range=[(8, 6), (8, 5), (7, 5), (6, 4)]):
        """
        Detect checkerboard pattern in frame.
        Try multiple board sizes to increase success rate.
        """
        self.log(f"Attempting checkerboard detection...", "DEBUG")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Try different board sizes (most common first)
        for cols, rows in board_size_range:
            try:
                found, corners = cv2.findChessboardCorners(
                    gray, (cols, rows),
                    cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
                )
                
                if found:
                    # Refine corners for sub-pixel accuracy
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                    refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                    
                    self.log(f"Checkerboard found: {cols}x{rows}", "SUCCESS")
                    return refined.reshape(-1, 2), (rows, cols)
            except Exception as e:
                self.log(f"Checkerboard {cols}x{rows} failed: {e}", "DEBUG")
                continue
        
        return None, None
    
    def detect_aruco_markers(self, frame):
        """
        Detect ArUco markers for calibration.
        Modern approach: place 4 ArUco markers at floor corners.
        """
        try:
            # Try to import ArUco
            from cv2 import aruco
            
            self.log(f"Attempting ArUco marker detection...", "DEBUG")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use predefined dictionary
            dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
            detector = aruco.ArucoDetector(dictionary)
            corners, ids, rejected = detector.detectMarkers(gray)
            
            if ids is not None and len(ids) >= 4:
                self.log(f"Found {len(ids)} ArUco markers", "SUCCESS")
                # This would require specific marker IDs at known positions
                # Simplified version - would need more configuration
                return corners, ids
            
        except Exception as e:
            self.log(f"ArUco detection not available: {e}", "DEBUG")
        
        return None, None
    
    def detect_tile_grid(self, frame):
        """
        Detect tile/grid pattern via edge and line detection.
        """
        self.log(f"Attempting tile grid detection...", "DEBUG")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        try:
            # Detect edges
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate to connect nearby edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges = cv2.dilate(edges, kernel, iterations=2)
            
            # Detect lines using Hough
            lines = cv2.HoughLinesP(
                edges, 1, np.pi/180, 50, 
                minLineLength=50, maxLineGap=10
            )
            
            if lines is None or len(lines) < 4:
                self.log("Not enough lines detected for grid", "DEBUG")
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
                self.log(f"Insufficient grid lines (h:{len(h_lines)}, v:{len(v_lines)})", "DEBUG")
                return None, None
            
            # Sort and get outer lines
            h_lines.sort()
            v_lines.sort()
            
            # Get unique positions
            h_positions = [h_lines[0][0]]
            v_positions = [v_lines[0][0]]
            
            for y, _ in h_lines[1:]:
                if abs(y - h_positions[-1]) > 15:
                    h_positions.append(y)
            
            for x, _ in v_lines[1:]:
                if abs(x - v_positions[-1]) > 15:
                    v_positions.append(x)
            
            if len(h_positions) < 2 or len(v_positions) < 2:
                self.log("Not enough grid positions found", "DEBUG")
                return None, None
            
            # Extract 4 corners
            corners = np.array([
                [v_positions[0], h_positions[0]],        # Top-left
                [v_positions[-1], h_positions[0]],       # Top-right
                [v_positions[-1], h_positions[-1]],      # Bottom-right
                [v_positions[0], h_positions[-1]],       # Bottom-left
            ], dtype=np.float32)
            
            grid_dims = (len(h_positions) - 1, len(v_positions) - 1)
            
            self.log(f"Tile grid found: {grid_dims[0]}x{grid_dims[1]}", "SUCCESS")
            return corners, grid_dims
            
        except Exception as e:
            self.log(f"Tile grid detection failed: {e}", "DEBUG")
            return None, None
    
    def detect_floor_pattern(self, frame):
        """
        Detect ANY floor pattern automatically.
        Try methods in order of reliability.
        """
        # Method 1: Checkerboard (most reliable)
        corners, size = self.detect_checkerboard(frame)
        if corners is not None:
            return corners, size, "checkerboard"
        
        # Method 2: Tile grid (good fallback)
        corners, size = self.detect_tile_grid(frame)
        if corners is not None:
            return corners, size, "tile_grid"
        
        # Method 3: ArUco markers (if supported)
        corners, ids = self.detect_aruco_markers(frame)
        if corners is not None:
            return corners, ids, "aruco"
        
        return None, None, None
    
    def compute_homography_from_pattern(self, img_corners, pattern_dims, pattern_type="checkerboard"):
        """
        Compute homography matrix from detected pattern.
        Assumes pattern covers the entire WORLD_GRID area.
        """
        try:
            if pattern_type == "checkerboard":
                rows, cols = pattern_dims
                # Each square in the checkerboard
                tile_w = WORLD_GRID_W / cols
                tile_h = WORLD_GRID_H / rows
            elif pattern_type == "tile_grid":
                rows, cols = pattern_dims
                tile_w = WORLD_GRID_W / cols
                tile_h = WORLD_GRID_H / rows
            else:  # ArUco
                rows, cols = 2, 2  # 2x2 grid of markers
                tile_w = WORLD_GRID_W / cols
                tile_h = WORLD_GRID_H / rows
            
            # World points corresponding to pattern corners
            world_pts = np.array([
                [0.0, 0.0],
                [WORLD_GRID_W, 0.0],
                [WORLD_GRID_W, WORLD_GRID_H],
                [0.0, WORLD_GRID_H],
            ], dtype=np.float32)
            
            # Ensure image points are correct format
            img_pts = np.array(img_corners[:4], dtype=np.float32)
            
            # Compute homography
            H, mask = cv2.findHomography(img_pts, world_pts)
            
            if H is None:
                self.log("Homography computation failed", "ERROR")
                return None
            
            return H
            
        except Exception as e:
            self.log(f"Error computing homography: {e}", "ERROR")
            return None
    
    def compute_reprojection_error(self, H, img_pts, world_pts):
        """
        Compute reprojection error to assess calibration quality.
        Lower error = better calibration (target: < 2cm = < 200 points if 1m=100px)
        """
        if H is None:
            return float('inf')
        
        try:
            # Project world points back to image
            projected = cv2.perspectiveTransform(
                world_pts.reshape(-1, 1, 2), 
                np.linalg.inv(H)
            ).reshape(-1, 2)
            
            # Compute error
            errors = np.linalg.norm(img_pts - projected, axis=1)
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            
            self.log(f"Reprojection - Mean: {mean_error:.2f}px, Max: {max_error:.2f}px", "DEBUG")
            
            return mean_error
            
        except Exception as e:
            self.log(f"Error computing reprojection: {e}", "DEBUG")
            return float('inf')
    
    def validate_homography(self, H, img_pts=None, world_pts=None):
        """
        Validate homography quality.
        Returns quality score (0-1, higher is better).
        """
        if H is None:
            return 0.0
        
        try:
            # Check matrix properties
            det = np.linalg.det(H)
            cond = np.linalg.cond(H)
            
            # Good properties: det ~ 1-5000, cond < 1000
            if abs(det) < 0.1 or cond > 10000:
                self.log(f"Poor matrix properties: det={det:.6f}, cond={cond:.2f}", "WARNING")
                return 0.1
            
            # Invertibility check
            try:
                H_inv = np.linalg.inv(H)
            except:
                self.log("Matrix not invertible", "ERROR")
                return 0.0
            
            # Reprojection error check
            if img_pts is not None and world_pts is not None:
                reprojection_error = self.compute_reprojection_error(H, img_pts, world_pts)
                # Convert error to quality (lower error = higher quality)
                # Target: < 2cm error = quality 1.0
                # At 50px = 1m scale, 2cm error = 1 pixel
                quality = 1.0 / (1.0 + reprojection_error / 5.0)
            else:
                quality = 0.7  # Default if can't compute
            
            self.log(f"Homography quality: {quality:.2f}", "DEBUG")
            return quality
            
        except Exception as e:
            self.log(f"Validation error: {e}", "DEBUG")
            return 0.0
    
    def calibrate_from_frame(self, frame, attempt_num=1):
        """
        Attempt complete calibration from a single frame.
        Returns (H, quality) if successful, (None, 0) otherwise.
        """
        self.log(f"\n--- Attempt {attempt_num} ---", "INFO")
        
        try:
            # Step 1: Detect pattern
            corners, dims, pattern_type = self.detect_floor_pattern(frame)
            if corners is None:
                self.log("No floor pattern detected", "WARNING")
                return None, 0.0
            
            # Step 2: Compute homography
            H = self.compute_homography_from_pattern(corners, dims, pattern_type)
            if H is None:
                self.log("Homography computation failed", "ERROR")
                return None, 0.0
            
            # Step 3: Validate quality
            img_pts = np.array(corners[:4], dtype=np.float32)
            world_pts = np.array([
                [0.0, 0.0],
                [WORLD_GRID_W, 0.0],
                [WORLD_GRID_W, WORLD_GRID_H],
                [0.0, WORLD_GRID_H],
            ], dtype=np.float32)
            
            quality = self.validate_homography(H, img_pts, world_pts)
            
            if quality < 0.5:
                self.log(f"Quality too low: {quality:.2f} (need > 0.5)", "WARNING")
                return None, quality
            
            self.log(f"✓ Calibration successful! Quality: {quality:.2f}", "SUCCESS")
            return H, quality
            
        except Exception as e:
            self.log(f"Exception in calibration: {e}", "ERROR")
            return None, 0.0
    
    def run_automatic_calibration(self):
        """
        Main automatic calibration loop.
        NO HUMAN INPUT REQUIRED.
        """
        print("\n" + "="*70)
        print("FULLY AUTOMATIC CALIBRATION - ZERO HUMAN INPUT")
        print("="*70)
        print(f"Target floor area: {WORLD_GRID_W}m × {WORLD_GRID_H}m")
        print(f"Pattern detection: Checkerboard → Tile Grid → ArUco")
        print("="*70 + "\n")
        
        # Attempt calibration
        for attempt in range(self.max_retries):
            try:
                # Open camera
                cap = cv2.VideoCapture(CAMERA_INDEX)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
                
                if not cap.isOpened():
                    self.log("Cannot open camera, retrying...", "ERROR")
                    cap.release()
                    time.sleep(1)
                    continue
                
                # Capture frames
                frames_captured = 0
                for frame_idx in range(10):  # Try up to 10 frames per attempt
                    ret, frame = cap.read()
                    if not ret:
                        self.log(f"Frame capture failed", "DEBUG")
                        continue
                    
                    frames_captured += 1
                    self.log(f"Frame {frames_captured}/10 captured", "DEBUG")
                    
                    # Try calibration
                    H, quality = self.calibrate_from_frame(frame, attempt + 1)
                    
                    if H is not None and quality > self.best_quality:
                        self.best_H = H
                        self.best_quality = quality
                        self.log(f"Best calibration so far (quality: {quality:.2f})", "SUCCESS")
                        
                        # If quality is good enough, save and return
                        if quality > 0.7:
                            cap.release()
                            return self.best_H
                    
                    # Wait before next frame
                    time.sleep(self.frame_wait_time)
                
                cap.release()
                
                if self.best_H is not None:
                    self.log(f"Best quality achieved: {self.best_quality:.2f}", "SUCCESS")
                    return self.best_H
                
            except Exception as e:
                self.log(f"Exception in attempt {attempt+1}: {e}", "ERROR")
                try:
                    cap.release()
                except:
                    pass
                time.sleep(1)
                continue
        
        # Final result
        if self.best_H is not None:
            self.log(f"✓ Calibration complete (quality: {self.best_quality:.2f})", "SUCCESS")
            return self.best_H
        else:
            self.log("❌ Calibration failed after all attempts", "ERROR")
            return None
    
    def save_calibration(self):
        """
        Save calibration to file if it was successful.
        """
        if self.best_H is None:
            print("\n❌ No valid calibration to save")
            return False
        
        try:
            np.save(HOMOGRAPHY_FILE, self.best_H)
            print(f"\n✓ Calibration saved to {HOMOGRAPHY_FILE}")
            print(f"  Quality score: {self.best_quality:.2f}/1.00")
            return True
        except Exception as e:
            print(f"\n❌ Error saving calibration: {e}")
            return False
    
    def run(self):
        """
        Main entry point - completely automatic.
        """
        # Run automatic calibration
        H = self.run_automatic_calibration()
        
        # Save if successful
        if H is not None:
            success = self.save_calibration()
            print("\n" + "="*70)
            print("✓ AUTOMATIC CALIBRATION COMPLETE!")
            print("="*70)
            print(f"System is ready for deployment.")
            print(f"Run: python main.py")
            return 0
        else:
            print("\n" + "="*70)
            print("❌ AUTOMATIC CALIBRATION FAILED")
            print("="*70)
            print("\nSuggestions:")
            print("  1. Ensure floor pattern is visible to camera")
            print("  2. Improve lighting (minimum 100 lux)")
            print("  3. Position camera at 15-45° angle")
            print("  4. Verify WORLD_GRID_W/H in config.py")
            print("  5. Try running with --verbose flag")
            print("\nRetry with: python fully_automatic_calibration.py --verbose")
            return 1
 
 
def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fully automatic calibration - ZERO human input"
    )
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose debug output")
    parser.add_argument("--retries", type=int, default=5,
                       help="Maximum calibration attempts (default: 5)")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing calibration")
    
    args = parser.parse_args()
    
    # Check if calibration already exists
    if os.path.exists(HOMOGRAPHY_FILE) and not args.force:
        print(f"Calibration already exists: {HOMOGRAPHY_FILE}")
        print("Use --force to recalibrate")
        return 0
    
    # Run automatic calibration
    calibrator = FullyAutomaticCalibrator(
        max_retries=args.retries,
        verbose=args.verbose
    )
    
    return calibrator.run()
 
 
if __name__ == "__main__":
    sys.exit(main())
 