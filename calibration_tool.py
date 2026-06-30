#!/usr/bin/env python3
"""
calibration_tool.py — FULLY AUTOMATIC Ground Plane Calibration Tool

This tool automatically detects floor patterns and calibrates with ZERO user interaction:
  • Automatic checkerboard detection
  • Automatic tile grid detection
  • Homography computation
  • Real-world scale estimation
  • Optional validation

Usage:
  python calibration_tool.py               — Automatic calibration
  python calibration_tool.py --validate    — Validate existing calibration
  python calibration_tool.py --recalibrate — Force automatic recalibration
"""

import sys
import argparse
import calibration
import cv2
import numpy as np
import os
from config import CAMERA_INDEX, CAPTURE_W, CAPTURE_H, HOMOGRAPHY_FILE, WORLD_GRID_W, WORLD_GRID_H


def print_banner():
    """Display welcome banner."""
    print("\n" + "="*70)
    print("  CROWD MONITOR — Ground Plane Calibration Tool")
    print("  FULLY AUTOMATIC (ZERO USER INTERACTION)")
    print("="*70 + "\n")


def interactive_mode():
    """Main automatic calibration workflow."""
    print_banner()
    print("STEP 1: Pre-Calibration Check")
    print("-" * 70)
    
    if calibration.is_calibrated():
        print("✓ Existing calibration found.")
        choice = input("  [A] Use existing | [B] Validate | [C] Recalibrate: ").strip().upper()
        if choice == "A":
            print("\n→ Using existing calibration. Exiting.")
            return True
        elif choice == "B":
            print("\nValidating calibration...")
            return validate_mode()
        elif choice == "C":
            print("\nRecalibrating (automatic pattern detection)...")
            return calibrate_mode()
        else:
            print("Invalid choice.")
            return False
    else:
        print("✗ No calibration file found.")
        print("\nAutomatic calibration starting...")
        print("Requirements:")
        print("  • Camera pointing at patterned floor")
        print("  • Checkerboard OR tile grid visible")
        print("  • Pattern should span most of camera view")
        print("  • Good lighting on floor\n")
        input("Press ENTER to begin automatic detection...")
        return calibrate_mode()


def calibrate_mode():
    """Perform automatic calibration."""
    print("\n" + "="*70)
    print("STEP 2: Automatic Pattern Detection")
    print("-" * 70)
    
    try:
        # Test camera connection
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
        
        if not cap.isOpened():
            print("✗ Cannot open camera. Check camera index in config.py")
            return False
        
        ret, test_frame = cap.read()
        cap.release()
        
        if not ret:
            print("✗ Cannot read from camera.")
            return False
        
        print("✓ Camera connected successfully")
        print(f"✓ Frame size: {CAPTURE_W}×{CAPTURE_H}")
        print(f"✓ Monitored area: {WORLD_GRID_W}m × {WORLD_GRID_H}m (from config)\n")
        
        # Automatic pattern detection and calibration
        H = calibration.auto_calibrate(frame=test_frame)
        
        if H is None:
            print("\n✗ Automatic calibration failed.")
            print("\nTroubleshooting:")
            print("  • Ensure floor has visible pattern (checkerboard or tiles)")
            print("  • Make sure pattern spans most of camera view")
            print("  • Try improving lighting")
            print("  • Manual calibration not available in this version")
            return False
        
        # Load the saved homography
        calibration.load_homography()
        
        print("\n" + "="*70)
        print("STEP 3: Validation")
        print("-" * 70)
        print("Grid overlay will be displayed. Check if it aligns with floor.\n")
        
        input("Press ENTER to view validation grid...")
        
        if calibration.validate_calibration(test_frame=test_frame):
            print("\n✓ Calibration validation: PASSED")
            print("\n" + "="*70)
            print("✓✓✓ AUTOMATIC CALIBRATION COMPLETE ✓✓✓")
            print("="*70)
            print("You can now run: python main.py\n")
            return True
        else:
            print("\n✗ Calibration validation: REJECTED")
            print("Removing invalid calibration file...")
            if os.path.exists(HOMOGRAPHY_FILE):
                os.remove(HOMOGRAPHY_FILE)
            print("Please try again with better floor visibility.\n")
            return False
            
    except KeyboardInterrupt:
        print("\n✗ Calibration cancelled by user.")
        return False
    except Exception as e:
        print(f"\n✗ Error during calibration: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_mode():
    """Validate existing calibration."""
    print("\n" + "="*70)
    print("VALIDATION: Existing Calibration")
    print("-" * 70)
    
    if not calibration.is_calibrated():
        print("✗ No calibration to validate.")
        return False
    
    try:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_H)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("✗ Cannot read from camera.")
            return False
        
        print("Grid overlay will be displayed. Check if it aligns with floor.")
        input("Press ENTER to continue...")
        
        if calibration.validate_calibration(test_frame=frame):
            print("\n✓ Calibration validation: PASSED")
            return True
        else:
            print("\n✗ Calibration validation: REJECTED")
            return False
            
    except Exception as e:
        print(f"✗ Validation error: {e}")
        return False


def recalibrate_mode():
    """Force automatic recalibration."""
    print("\n" + "="*70)
    print("RECALIBRATION (Automatic Pattern Detection)")
    print("-" * 70)
    
    if not calibration.is_calibrated():
        print("✗ No existing calibration to replace.")
        return calibrate_mode()
    
    print("This will replace your current calibration with new automatic detection.")
    confirm = input("Are you sure? (yes/no): ").strip().lower()
    
    if confirm == "yes":
        return calibrate_mode()
    else:
        print("✗ Recalibration cancelled.")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ground Plane Calibration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calibration_tool.py               # Interactive mode
  python calibration_tool.py --validate    # Validate calibration
  python calibration_tool.py --recalibrate # Force recalibration
        """
    )
    parser.add_argument("--validate", action="store_true", 
                        help="Validate existing calibration")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Force recalibration")
    
    args = parser.parse_args()
    
    try:
        if args.validate:
            success = validate_mode()
        elif args.recalibrate:
            success = recalibrate_mode()
        else:
            success = interactive_mode()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\nCalibration tool interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
