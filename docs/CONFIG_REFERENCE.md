# Configuration Reference — Ground Plane Calibration

## Critical: Pre-Calibration Settings

### Step 1: Measure Your Monitored Area

**On your floor, measure:**
- Total width (left to right): _____ metres
- Total depth (top to bottom): _____ metres

Example: A school hallway
```
     ←─── 10 metres ───→
   ┌─────────────────────┐
   │  Monitored Area     │ 8 metres
   │                     │ (depth)
   │    (Camera View)    │
   └─────────────────────┘
```

### Step 2: Update config.py

```python
# BEFORE (default - change these!)
WORLD_GRID_W = 10.0     # Total floor WIDTH (metres)
WORLD_GRID_H = 8.0      # Total floor HEIGHT/DEPTH (metres)

# AFTER (your measurements)
WORLD_GRID_W = 12.0     # Update to your width
WORLD_GRID_H = 10.0     # Update to your depth
```

### Step 3: Find/Mark Reference Rectangle

Choose a rectangular area on your floor that:
- Has clear, visible corners
- Is ≥2 metres × ≥2 metres
- Fits within your monitored area
- Can be measured with tape measure

Example locations:
- Tile grid intersections
- Parking space corners  
- Taped rectangle
- Door frame edges

**Measure this rectangle:**
- Width: _____ metres
- Height: _____ metres

---

## Full config.py Reference

```python
# ────────────────────────────────────────────────────────────────────
# CAMERA CONFIGURATION
# ────────────────────────────────────────────────────────────────────

CAMERA_INDEX    = 0         # USB camera: 0=default, 1=second, etc.
CAPTURE_W       = 1280      # Capture width (pixels)
CAPTURE_H       = 720       # Capture height (pixels)
DISPLAY_W       = 1600      # Display width (pixels)
DISPLAY_H       = 900       # Display height (pixels)

# Notes:
# • CAPTURE_W/H should be 16:9 aspect ratio
# • DISPLAY_W/H should match your monitor
# • If camera doesn't open, try CAMERA_INDEX = 1 or 2


# ────────────────────────────────────────────────────────────────────
# GROUND PLANE CALIBRATION ⭐ SET THESE FIRST!
# ────────────────────────────────────────────────────────────────────

HOMOGRAPHY_FILE = "homography.npy"  # Where calibration is saved
                                     # Don't change this!

WORLD_GRID_W    = 10.0      # ⭐ Total floor WIDTH you monitor (metres)
WORLD_GRID_H    = 8.0       # ⭐ Total floor HEIGHT/DEPTH (metres)

# ⚠️  IMPORTANT: Set these to match your actual floor area
#     These define the extent of the grid overlay
#     Too small = grid compressed
#     Too large = grid spread out
#     Must be set BEFORE calibration!

# How to determine:
# 1. Measure your monitored floor area with tape measure
# 2. Width = left to right (WORLD_GRID_W)
# 3. Height = top to bottom or front to back (WORLD_GRID_H)
# 4. Set these values in config.py
# 5. THEN run calibration


# ────────────────────────────────────────────────────────────────────
# GRID CONFIGURATION
# ────────────────────────────────────────────────────────────────────

GRID_ROWS       = 4         # Number of rows in analysis grid
GRID_COLS       = 4         # Number of columns in analysis grid

# Notes:
# • 4×4 = 16 cells total
# • Each cell analyzed independently for density
# • Don't change unless you have specific needs


# ────────────────────────────────────────────────────────────────────
# FALLBACK AREA (used only when NOT calibrated)
# ────────────────────────────────────────────────────────────────────

CELL_AREA_M2    = 2.0       # Uniform area per grid cell (m²)
                             # Used only if homography.npy missing

# How to estimate CELL_AREA_M2:
# 1. Stand in camera view
# 2. Identify one grid cell on the floor
# 3. Measure that cell with tape measure
#    width × depth = area
# 4. Example: 1.6m × 1.25m = 2.0 m²


# ────────────────────────────────────────────────────────────────────
# DETECTION CONFIGURATION
# ────────────────────────────────────────────────────────────────────

MODEL_PATH      = "yolov8m.pt"  # YOLO model path
CONF_THRESHOLD  = 0.25          # Detection confidence threshold
BUFFER_SIZE     = 15            # Temporal smoothing (frames)


# ────────────────────────────────────────────────────────────────────
# OCCLUSION CORRECTION (handles crowded scenes)
# ────────────────────────────────────────────────────────────────────

OCCLUSION_CORRECTION = True     # Enable occlusion correction
OCCLUSION_GAIN       = 0.15     # Correction strength (0-1)

# Note: Papers show YOLO recall drops ~42% in dense crowds
#       This correction factor partially compensates


# ────────────────────────────────────────────────────────────────────
# DENSITY THRESHOLDS (Polus et al. 1983 standards)
# ────────────────────────────────────────────────────────────────────

DENSITY_THRESHOLDS = [
    (0.50,  "VERY LOW",  (80,  220, 100)),   # Green
    (0.80,  "LOW",       (0,   220, 255)),   # Cyan
    (1.50,  "MODERATE",  (0,   200, 255)),   # Light blue
    (3.00,  "HIGH",      (0,   140, 255)),   # Orange-ish
    (6.00,  "CRITICAL",  (60,  60,  255)),   # Red
]

# Thresholds are in people per m²
# These are industry-standard from Polus, Schofer & Ushpiz (1983)
```

---

## Calibration Workflow

```
┌─────────────────────────────────────────────────────┐
│  1. MEASUREMENT PHASE                              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ✓ Measure floor area with tape measure            │
│  ✓ Update WORLD_GRID_W and WORLD_GRID_H            │
│  ✓ Mark reference rectangle (4 corners)            │
│  ✓ Measure reference rectangle dimensions          │
│                                                     │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│  2. CALIBRATION PHASE                              │
├─────────────────────────────────────────────────────┤
│                                                     │
│  $ python calibration_tool.py                      │
│  • Opens camera                                     │
│  • Shows live preview                              │
│  • Click 4 corners: TL → TR → BR → BL             │
│  • Enter reference rectangle dimensions            │
│  • System computes homography matrix H             │
│                                                     │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│  3. VALIDATION PHASE                               │
├─────────────────────────────────────────────────────┤
│                                                     │
│  • Grid overlay shown on camera                    │
│  • Green circles at 4 reference corners            │
│  • Press SPACE to accept                           │
│  • Press ESC to recalibrate                        │
│                                                     │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│  4. SAVED                                           │
├─────────────────────────────────────────────────────┤
│                                                     │
│  File: homography.npy (3×3 matrix)                 │
│  → Loaded at every startup                        │
│  → Used for px_to_world transforms                │
│                                                     │
└──────────────────┬──────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────┐
│  5. MONITORING READY                               │
├─────────────────────────────────────────────────────┤
│                                                     │
│  $ python main.py                                  │
│  ✓ homography.npy loaded                          │
│  ✓ Detections have world coordinates (wx, wy)     │
│  ✓ Density is perspective-correct                 │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Detection Output Format

### Before Calibration

```python
detection = {
    'x1': 100,      # Pixel bbox left
    'y1': 200,      # Pixel bbox top
    'x2': 150,      # Pixel bbox right
    'y2': 280,      # Pixel bbox bottom
    'cx': 125,      # Pixel centroid x
    'cy': 240,      # Pixel centroid y
    'wx': None,     # World x (not calibrated)
    'wy': None,     # World y (not calibrated)
    'row': 1,       # Grid row
    'col': 2,       # Grid col
    'pid': 42       # Track ID
}
```

### After Calibration

```python
detection = {
    'x1': 100,      # Pixel bbox left
    'y1': 200,      # Pixel bbox top
    'x2': 150,      # Pixel bbox right
    'y2': 280,      # Pixel bbox bottom
    'cx': 125,      # Pixel centroid x
    'cy': 240,      # Pixel centroid y
    'wx': 2.5,      # World x (metres) ← CALIBRATED!
    'wy': 1.8,      # World y (metres) ← CALIBRATED!
    'row': 1,       # Grid row
    'col': 2,       # Grid col
    'pid': 42       # Track ID
}
```

---

## Grid Cell Area Calculation

### Uncalibrated (Fallback)

```python
# All cells: same constant area
for row in range(GRID_ROWS):
    for col in range(GRID_COLS):
        cell_area_m2 = CELL_AREA_M2  # 2.0 everywhere ❌
        
# Problem: Wrong for angled cameras!
```

### Calibrated (Perspective-Corrected)

```python
# Each cell: different area based on perspective
for row in range(GRID_ROWS):
    for col in range(GRID_COLS):
        cell_area_m2 = calibration.cell_area_m2(row, col)
        # Result: Varies from 2.2 to 1.7 depending on position ✓
```

---

## Coordinate System

### World Space Origin

```
Camera View (perspective)
┌──────────────────────────┐
│ (0,0)▲ (W,0)             │
│   ┌──────────────────┐   │
│   │   World Grid     │   │
│   │                  │   │
│   │                  │   │
│   └──────────────────┘   │
│ (0,H)              (W,H) │
└──────────────────────────┘

World Space:
• Origin (0, 0) = top-left corner of reference rectangle
• X-axis → rightward (metres)
• Y-axis → downward (metres)
• Units: metres
• Range: [0, WORLD_GRID_W] × [0, WORLD_GRID_H]
```

---

## Validation Checklist

After setting config.py and before calibrating:

- [ ] WORLD_GRID_W set to actual floor width (metres)
- [ ] WORLD_GRID_H set to actual floor depth (metres)
- [ ] CAMERA_INDEX correct (camera opens successfully)
- [ ] Reference rectangle marked on floor
- [ ] Reference rectangle dimensions measured
- [ ] CELL_AREA_M2 estimated (backup value)
- [ ] homography.npy doesn't exist yet (fresh start)

---

## Common Config Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| WORLD_GRID_W too small | Grid compressed in image | Measure floor, increase value |
| WORLD_GRID_H too large | Grid spread out | Measure floor, decrease value |
| CAMERA_INDEX wrong | "Cannot open camera" | Try 0, 1, 2... until works |
| CAPTURE_W/H not 16:9 | Image distorted | Use 1280×720 or 1920×1080 |
| CELL_AREA_M2 = 0 | Division by zero crash | Set to reasonable value (1-5) |

---

## Quick Setup Checklist

```
□ Step 1: Measure Floor
  └─ Width: _____ m, Height: _____ m

□ Step 2: Update config.py
  └─ WORLD_GRID_W = _____ 
  └─ WORLD_GRID_H = _____

□ Step 3: Mark Reference Rectangle
  └─ Mark 4 corners on floor
  └─ Measure: Width _____ m, Height _____ m

□ Step 4: Run Calibration
  └─ python calibration_tool.py
  └─ Click 4 points (TL, TR, BR, BL)
  └─ Enter rectangle dimensions
  └─ Validate grid overlay

□ Step 5: Start Monitoring
  └─ python main.py
  └─ ✓ Verify homography.npy loaded
  └─ ✓ Check detections have wx, wy

□ Done!
  └─ Press C during monitoring to validate/recalibrate
```

---

**Last Updated:** June 2026  
**Status:** Ready for Production
