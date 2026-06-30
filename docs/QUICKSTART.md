# Ground Plane Calibration — Quick Reference

## TL;DR — Start Here

### First Time Setup
```bash
python calibration_tool.py
# Follow 5-step wizard
# Creates: homography.npy
```

### During Monitoring
```
Press C to:
  • Validate current calibration
  • Recalibrate if camera moved
  • Check calibration quality
```

---

## The Problem This Solves

**Without calibration:**
- Crowd density near camera looks 3-4x more dense than it actually is
- Density at far end of scene is underestimated  
- Results depend on camera angle (angled wall camera? Meaningless)

**With calibration:**
- All density measurements are physically accurate
- Works with ANY camera angle (overhead, wall-mounted, angled)
- Perspective distortion is automatically corrected

---

## What You Need

1. A **tape measure** (1x)
2. A **clear rectangular area on the floor** (≥2m×2m)
3. This area **must be visible** from your camera

Example:
```
Mark on floor:
  • 4 corners of a rectangle
  • Width: 4.5 metres
  • Height: 3.0 metres
```

---

## 3-Step Workflow

### Step 1: Prepare Reference Rectangle
```
Mark 4 corners on the floor in a rectangle shape.
Measure width and height with tape measure.
```

### Step 2: Run Calibration Tool
```bash
python calibration_tool.py
```
- Click 4 corners in order: TL → TR → BR → BL
- Enter width and height
- Validation preview
- File saved: homography.npy

### Step 3: Start Monitoring
```bash
python main.py
```
System automatically loads homography.npy.
You will be asked if you want to use AI auto-segmentation for the walkable area.
All coordinates are now perspective-corrected and the UI will flash congested cells!

---

## Keyboard Controls

### During Monitoring
| Key | Action |
|-----|--------|
| `Q` | Quit monitoring |
| `C` | Open calibration menu |

### In Calibration Menu
| Key | Action |
|-----|--------|
| `V` | Validate calibration |
| `R` | Recalibrate |
| `Q` | Return to monitoring |

---

## Configuration (config.py)

These MUST be set correctly:

```python
# Camera
CAMERA_INDEX = 0          # USB camera ID
CAPTURE_W    = 1280       # Resolution width
CAPTURE_H    = 720        # Resolution height

# Monitored area (SET BEFORE CALIBRATING!)
WORLD_GRID_W = 10.0       # Total floor width you monitor (metres)
WORLD_GRID_H = 8.0        # Total floor depth you monitor (metres)

# Homography file (don't change)
HOMOGRAPHY_FILE = "homography.npy"

# Fallback (used only if not calibrated)
CELL_AREA_M2 = 2.0        # Uniform area per grid cell (m²)
```

---

## Common Issues & Fixes

| Issue | Fix |
|-------|-----|
| "Homography computation failed" | Re-click 4 points with better spacing |
| Grid overlay misaligned | Points weren't at exact corners; recalibrate |
| Density still looks wrong | Check WORLD_GRID_W/H match your floor extent |
| Can't open camera | Check CAMERA_INDEX in config.py |
| Calibration disappeared | File system error; recalibrate |

---

## Files Created/Modified

| File | Status | Purpose |
|------|--------|---------|
| `calibration_tool.py` | ✨ NEW | Interactive calibration GUI |
| `calibration.py` | 🔄 ENHANCED | Live preview, validation |
| `detector.py` | 🔄 ENHANCED | Now computes wx, wy per detection |
| `main.py` | 🔄 ENHANCED | Recalibration menu, keyboard controls |
| `CALIBRATION_GUIDE.md` | 📖 NEW | Full documentation |
| `homography.npy` | 💾 AUTO | Binary matrix (don't edit) |

---

## Data Flow

```
Camera Frame
    ↓
[DETECTION] (YOLO)
    ↓
For each person:
  pixel coords (cx, cy)
        ↓
  [HOMOGRAPHY TRANSFORM]
        ↓
  world coords (wx, wy)
        ↓
[DENSITY CALCULATION]
(using real m² areas)
        ↓
Perspective-correct density
```

---

## Detection Output (detector.py)

Each detection now includes:

```python
{
  'cx': 125, 'cy': 240,    # Pixel coordinates
  'wx': 2.5, 'wy': 1.8,    # World coordinates (metres) ← NEW!
  'x1': 100, 'y1': 200,    # Bounding box (pixels)
  'x2': 150, 'y2': 280,
  'row': 1, 'col': 2,      # Grid cell
  'pid': 42                 # Tracking ID
}
```

If NOT calibrated: `wx=None, wy=None`

---

## API Cheat Sheet

```python
import calibration

# Status
calibration.is_calibrated()           # → bool

# Transform
calibration.px_to_world([(100, 200)]) # → [[2.5, 1.8]]
calibration.world_to_px([[2.5, 1.8]]) # → [[100, 200]]

# Grid
calibration.cell_area_m2(row=1, col=2) # → 1.95 m²
calibration.world_to_grid(2.5, 1.8)    # → (row=1, col=2)

# Interactive
calibration.validate_calibration()     # Show grid overlay
calibration.recalibrate_interactive()  # Re-calibrate
```

---

## Validation Checklist

After calibration, verify:

- [ ] Green circles appear at 4 corners of reference rectangle
- [ ] Grid lines roughly align with floor boundaries
- [ ] No obvious distortion or misalignment
- [ ] homography.npy file exists
- [ ] System recognizes as calibrated: `✓ Calibrated`

---

## Example: Step-by-Step

### Day 1: Initial Setup
```bash
# 1. Measure floor area
# → 5m wide × 4m deep

# 2. Mark 4 corners with tape

# 3. Set config.py
WORLD_GRID_W = 5.0
WORLD_GRID_H = 4.0

# 4. Run calibration
python calibration_tool.py
# → Click 4 points
# → Enter: 5.0, 4.0
# → Validate ✓
# → homography.npy saved

# 5. Start monitoring
python main.py
# → ✓ Homography loaded
# → Monitoring active
```

### Day 2: Regular Monitoring
```bash
python main.py
# → Automatically loads homography.npy
# → All density measurements are accurate
# → Press C if you need to recalibrate
```

### If Camera Moves
```
During monitoring, press C:
  [R] Recalibrate
# → New homography computed
# → Monitoring resumes immediately
```

---

## Performance

| Operation | Time |
|-----------|------|
| Calibration | ~2 minutes |
| Per-frame transform | <1ms |
| Validation | <2 seconds |
| Recalibration | ~2 minutes |

**Zero performance impact on monitoring.**

---

## Under the Hood (For Nerds 🤓)

The system uses a **3×3 homography matrix H**:

```
Transformation:   world_pt = H @ pixel_pt (homogeneous)

Why it works:
• Maps 2D pixels to 2D ground plane
• Handles perspective distortion
• Computed from 4 reference points
• Invertible (pixel ← → world)

Math:
  H = cv2.findHomography(img_pts, world_pts)
  world = cv2.perspectiveTransform(pixel, H)
```

---

## Troubleshooting Tree

```
Calibration failed?
├─ "Computation failed"
│  └─ Re-click 4 points with better spacing
│
├─ "Can't open camera"
│  └─ Check CAMERA_INDEX in config.py
│
└─ "Grid misaligned at validation"
   └─ Points weren't at exact corners → recalibrate

Density still wrong after calibration?
├─ WORLD_GRID_W/H not set correctly?
│  └─ Update config.py with actual floor extent
│
├─ Detections don't have wx/wy?
│  └─ Check detector.py is updated
│
└─ Camera physically moved?
   └─ Press C → R to recalibrate
```

---

## Tips & Tricks

1. **Mark reference rectangle with tape** on the floor for clarity
2. **Use natural markers** (tile grout, floor seams) if possible  
3. **Recalibrate annually** or after any camera adjustment
4. **Validate monthly** with `Press C → V`
5. **Test with known crowd** to verify accuracy

---

## Need Help?

1. Read full guide: `CALIBRATION_GUIDE.md`
2. Check your configuration: `config.py`
3. Test calibration: `python calibration_tool.py --validate`
4. Debug detections: Add print statement in main loop
5. Recalibrate: Press `C` during monitoring → `R`

---

**Version 2.0 — Ground Plane Mapping & Live Calibration**
