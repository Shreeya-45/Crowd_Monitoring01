# Ground Plane Mapping & Homography Calibration Guide

## Overview

This guide explains the enhanced ground-plane calibration system for your crowd monitoring platform. The system uses **homography transformation** to map camera pixels to real-world floor coordinates, enabling perspective-correct density measurements regardless of camera angle.

---

## What's New ✨

### 1. **Live Camera Calibration with Real-time Preview**
- Interactive 4-point calibration UI
- Live grid overlay visualization
- Reprojection error metrics
- Validation feedback

### 2. **World Coordinate Integration**
- All detections now include `wx, wy` (world coordinates in metres)
- Per-cell real-world area calculation
- Perspective-correct density measurements

### 3. **Enhanced Tooling**
- **`calibration_tool.py`** — Standalone interactive calibration GUI
- **Runtime recalibration** — Press `C` during monitoring to recalibrate
- **Validation system** — Check calibration quality anytime

### 4. **Improved Main Loop**
- Graceful fallback to uniform constant if not calibrated
- Keyboard controls for calibration menu
- Better status feedback

---

## How It Works

### The Homography Matrix (H)

A **3×3 homography matrix** is a linear transformation that maps 2D pixel coordinates to 2D world (ground-plane) coordinates:

```
[  h11  h12  h13  ] 
H = [  h21  h22  h23  ]
[  h31  h32  h33  ]

world_pt = H @ pixel_pt  (in homogeneous coordinates)
```

**Key properties:**
- Handles perspective distortion
- Works with ANY camera angle (not just overhead)
- Computed from 4 reference points on the floor
- Corrects non-linear scale variation across the frame

### Why It Matters

**Without calibration:** 
- Single `CELL_AREA_M2 = 2.0` constant applies everywhere
- Objects near camera occupy more pixels → density overcount
- Objects far from camera occupy fewer pixels → density undercount
- Results are physically inaccurate for angled cameras

**With calibration:**
- Each pixel's real-world position is known
- Hull areas computed directly in m²
- Grid cells have correct real-world areas
- Density is perspective-correct

---

## Quick Start

### Option A: Interactive Calibration During Startup

```bash
python main.py
```

When prompted:
```
[CALIBRATION CHECK]
✗ No calibration file found.

  Option 1: Calibrate now (recommended, takes ~2 minutes)
  Option 2: Continue with fallback constant (less accurate)
  Option 3: Run calibration_tool.py separately and restart

  Enter choice (1/2/3): 1
```

Then follow the on-screen prompts.

### Option B: Standalone Calibration Tool (Recommended)

```bash
python calibration_tool.py
```

**Advantages:**
- Step-by-step guided interface
- Better error messages
- Validation preview
- Recalibration options

### Option C: Runtime Recalibration

During monitoring, press `C`:
```
[CALIBRATION MENU]
Current status: ✓ Calibrated

Options:
  [V] Validate current calibration
  [R] Recalibrate
  [Q] Return to monitoring

Enter choice (V/R/Q): V
```

---

## The Calibration Process

### Step 1: Prepare Reference Rectangle

**You need:**
- A tape measure
- Clear floor space (≥2m × 2m recommended)
- Visible from camera angle

**Mark on floor:**
- A rectangular area with known dimensions
- Clear corners (ideally at ground level)

**Measure:**
- Width (left → right) in metres
- Height (top → bottom) in metres

Example: A 4m wide × 3m deep area
```
Enter dimensions:
  Width  (metres):  4.0
  Height (metres):  3.0
```

### Step 2: Click 4 Reference Points

The calibration UI will show a live camera frame:

```
Click floor point 1/4 (ESC cancel)
```

**In this exact order:**

1. **Top-left** corner of your reference rectangle
2. **Top-right** corner
3. **Bottom-right** corner
4. **Bottom-left** corner

**Tips:**
- Click at the exact corner point on the floor
- Be as precise as possible (±5cm error introduces ~1% distortion)
- If points are collinear or coincident, calibration will fail

```
Visual example:

Camera view
┌─────────────────────────────┐
│                             │
│  ①————————————————————————②  │
│  │                         │
│  │   Reference Rectangle   │
│  │      (measured)         │
│  │                         │
│  ④————————————————————————③  │
│                             │
└─────────────────────────────┘

Click in order: ① → ② → ③ → ④
```

### Step 3: Validation

A preview shows the computed grid overlaid on the camera view:

```
Calibration Validation — Grid Overlay
Green circles = world grid corners. Press SPACE to accept or ESC to recalibrate.
```

**Check:**
- Green circles should be at the 4 corners you clicked
- Grid lines should roughly align with floor boundaries
- If misaligned → press ESC and recalibrate with more accurate points

### Step 4: Saved

Homography matrix saved to:
```
homography.npy
```

**Never edit this file manually.** To change calibration, recalibrate using the tools.

---

## Configuration

### In `config.py`:

```python
# Camera settings (must match your actual camera)
CAMERA_INDEX    = 0        # USB camera ID (0 = default)
CAPTURE_W       = 1280     # Capture resolution
CAPTURE_H       = 720      # Must be 16:9 aspect ratio

# Monitoring area extent in world space
WORLD_GRID_W    = 10.0     # Total floor width covered (metres)
WORLD_GRID_H    = 8.0      # Total floor depth covered (metres)

# Fallback if not calibrated
CELL_AREA_M2    = 2.0      # Uniform area per grid cell (metres²)

# Homography file
HOMOGRAPHY_FILE = "homography.npy"
```

**Important:** Set `WORLD_GRID_W` and `WORLD_GRID_H` BEFORE calibrating. These define the total monitored area that the grid spans in world coordinates.

---

## Using Calibrated Coordinates

### In `detector.py`

Each detection now includes world coordinates:

```python
detection = {
    'x1': 100,      # pixel bounding box left
    'y1': 200,      # pixel bounding box top
    'x2': 150,      # pixel bounding box right
    'y2': 280,      # pixel bounding box bottom
    'cx': 125,      # pixel centroid x
    'cy': 240,      # pixel centroid y
    'wx': 2.5,      # world centroid x (metres)
    'wy': 1.8,      # world centroid y (metres)
    'row': 1,       # grid row
    'col': 2,       # grid col
    'pid': 42       # person tracking ID
}
```

### In `density.py`

Convex hull computed in world space:

```python
hull_area_m2, hull_pts_px, hull_pts_w = convex_hull_area_m2(detections)

# hull_area_m2  — Real floor area in m²
# hull_pts_px   — Pixel coords for drawing
# hull_pts_w    — World coords for analysis
```

### Direct Transformations

```python
import calibration

# Pixel → World
pixel_pts = [(100, 200), (150, 250)]
world_pts = calibration.px_to_world(pixel_pts)
# Result: [[2.5, 1.8], [3.1, 2.2]]

# World → Pixel
world_pts = [[2.5, 1.8], [3.1, 2.2]]
pixel_pts = calibration.world_to_px(world_pts)
# Result: [[100, 200], [150, 250]]

# Get real area of grid cell (row, col)
area_m2 = calibration.cell_area_m2(row=1, col=2)
# Result: 1.95 (per-cell corrected area)

# Check calibration status
if calibration.is_calibrated():
    print("Using perspective-corrected coordinates")
else:
    print("Using fallback uniform constant")
```

---

## Troubleshooting

### "Homography computation failed — check your 4 points"

**Causes:**
- Points are collinear (all on a line)
- Points form a very small quadrilateral
- Some points are too close together

**Fix:** Re-click with better-spaced points covering the full reference rectangle.

---

### Grid overlay doesn't align with floor

**Causes:**
- Reference points were not at exact corners
- Measurement error in width/height
- Camera fisheye distortion

**Fix:**
1. Press ESC at validation
2. Recalibrate with more precision
3. If persistent, camera may need lens calibration (advanced)

---

### Density still looks wrong after calibration

**Check:**
1. Is `WORLD_GRID_W` and `WORLD_GRID_H` set correctly?
   ```python
   # config.py
   WORLD_GRID_W = 10.0  # Total monitored area width
   WORLD_GRID_H = 8.0   # Total monitored area depth
   ```

2. Are detections being assigned world coordinates?
   ```bash
   # Add debug output to main loop
   print(f"Detection: wx={d['wx']}, wy={d['wy']}")
   ```

3. Is the system actually calibrated?
   ```bash
   # Check during monitoring
   Press C → V (validate)
   ```

---

### Can I use a non-rectangular reference area?

**No.** The calibration requires a **rectangle** with horizontal/vertical edges. This is because:
- 4-point homography assumes planar surface
- Rectangle provides 2 orthogonal reference directions
- Non-rectangular shapes would need more points

**Workaround:** If your monitored area is non-rectangular, choose the largest axis-aligned rectangle within it.

---

## Advanced Usage

### Multi-Point Recalibration

For higher accuracy with distorted lenses, you can modify `calibration.py` to use 8-point calibration:

```python
# In run_calibration(), change:
if len(_clicked_pts) == 8:  # Instead of 4
    break
```

This requires clicking 8 points on a grid pattern.

### Recalibration Without Restart

```bash
# During monitoring, press 'C'
[CALIBRATION MENU]
  [R] Recalibrate
```

This updates `homography.npy` and reloads without stopping the system.

### Batch Calibration

For multiple cameras/angles:

```bash
# Each camera location gets its own calibration
python calibration_tool.py
# Saves to: homography.npy

# Rename for camera 1
mv homography.npy homography_cam1.npy

# Then recalibrate for camera 2
python calibration_tool.py
# Saves to: homography.npy

# In config.py, add logic to load correct file
```

---

## Performance Impact

- **Calibration process:** ~2 minutes (one-time)
- **Per-frame overhead:** <1ms (homography math is fast)
- **Memory:** ~1 MB (H matrix is 3×3, very small)

**No performance penalty for using calibrated coordinates vs. fallback.**

---

## API Reference

### `calibration.py` Functions

```python
# Load/Status
load_homography()              # Load H from disk, return H or None
is_calibrated()                # Return bool
get_homography()               # Return 3×3 H matrix
get_homography_inv()           # Return inverse 3×3 H^-1 matrix

# Transformations
px_to_world(pixel_pts)         # Nx2 pixels → Nx2 world metres
world_to_px(world_pts)         # Nx2 metres → Nx2 pixels

# Grid
cell_area_m2(row, col)         # Get real m² of cell (row, col)
world_to_grid(wx, wy)          # (wx, wy) → (row, col)
all_cell_areas()               # Return full [ROWS][COLS] area table

# UI
run_calibration(frame=None)    # Interactive 4-point calibration
validate_calibration(frame)    # Show and validate grid overlay
recalibrate_interactive()      # Recalibrate with confirmation
run_live_calibration_ui()      # Full interactive workflow
draw_world_grid(frame)         # Draw grid overlay on frame

# Inspection
get_calibration_quality()      # Return quality score 0–1 or None
```

---

## Best Practices

1. **Calibrate early** — Do this on first setup
2. **Measure accurately** — Use a tape measure, not estimates
3. **Choose a good rectangle** — Corners should be clearly visible
4. **Validate after calibration** — Check grid alignment before monitoring
5. **Recalibrate if camera moves** — Any physical repositioning invalidates H
6. **Set WORLD_GRID dimensions correctly** — Must match your monitored area

---

## Related Files

| File | Purpose |
|------|---------|
| `calibration.py` | Core homography & transformation math |
| `calibration_tool.py` | **NEW** — Interactive calibration GUI |
| `detector.py` | **UPDATED** — Now computes wx, wy per detection |
| `main.py` | **UPDATED** — Recalibration menu support |
| `density.py` | Uses world coordinates for hull area |
| `config.py` | WORLD_GRID_W/H, HOMOGRAPHY_FILE settings |

---

## Example: End-to-End Workflow

```bash
# 1. First time setup
python calibration_tool.py
# → Follow prompts, save homography.npy

# 2. Start monitoring
python main.py
# → System loads homography.npy automatically
# → Detections have wx, wy coordinates
# → Density is perspective-correct

# 3. During monitoring, check calibration
# → Press C to open calibration menu
# → Choose V to validate or R to recalibrate

# 4. If camera moves
# → Press C → R to recalibrate
# → homography.npy is updated
# → Monitoring continues with new calibration
```

---

## Frequently Asked Questions

**Q: What if I don't calibrate?**  
A: System falls back to uniform `CELL_AREA_M2` constant. Density will be inaccurate with angled cameras.

**Q: Can I calibrate with an overhead camera?**  
A: Yes! Overhead calibration still works. The homography will be identity-like (minimal perspective correction).

**Q: How precise must my reference rectangle be?**  
A: Within ±5 cm is typical. Larger errors cause proportional density errors.

**Q: Can I move the camera after calibration?**  
A: No. Any movement invalidates the homography. Recalibrate using the calibration menu (press C).

**Q: How long is the calibration valid?**  
A: Until the camera moves or is adjusted. Check occasionally with validation (press C → V).

**Q: What units are wx, wy in?**  
A: Metres. They represent absolute floor positions relative to your reference rectangle's (0,0) at top-left corner.

---

## Support

For issues:

1. Check the "Troubleshooting" section above
2. Run `python calibration_tool.py --validate` to check current calibration
3. Review reprojection errors displayed during calibration
4. Check that WORLD_GRID_W/H match your physical setup

---

**Last Updated:** June 2026  
**Version:** 2.0 (Ground Plane Mapping & Live Calibration)
