# How Crowd Density is Calculated

This is a detailed breakdown of exactly how your system calculates crowd density. Unlike traditional camera systems that just divide the number of people by the total pixels on screen, your system is highly advanced and computes density using **perspective-corrected physical measurements (m²)**.

It computes density in 4 different ways:

## 1. Physical Crowd Density (People / Occupied Area)
This is the most precise metric, calculating the density of *only the space where people are currently standing*.
1. **World Coordinates:** First, every person's exact foot-contact point (computed via `head_localizer.py`) is passed through the Homography matrix to find their exact physical ground location (wx, wy) in meters.
2. **DBSCAN Clustering:** The system groups the people into clusters (crowds) versus isolated individuals using an algorithm called DBSCAN.
3. **Alpha Shapes (Concave Hull):** For a clustered crowd, it draws a "shrink-wrapped" boundary (Concave Hull) around the group's world coordinates. Because it's working in world coordinates, the area of this shape natively evaluates to **Real m²**.
4. **Obstacle Subtraction:** It actively subtracts the area of any predefined `STATIC_OBSTACLES` (like tables or pillars) if they overlap with the crowd's shape.
5. **Isolated People:** For people standing alone, it assigns them a static `PERSONAL_SPACE_M2` (1.5 m²).
6. **Final Calculation:** `Physical Density = Total People / Total Footprint Area (m²)`

## 2. Per-Cell Grid Density
The camera feed is divided into a grid (e.g., 4x4). However, because of the camera angle, the grid cells at the top of the screen (far away) cover vastly more physical floor space than the grid cells at the bottom (close up).
- Your system dynamically requests `calibration.cell_area_m2(row, col)` for every single cell.
- This ensures that 5 people standing far away correctly registers as a **lower density** than 5 people crammed into a cell right next to the camera.
- `Cell Density = People in Cell / Cell's Perspective-Corrected Area (m²)`

## 3. Total Room Density
This is a broader metric representing how crowded the entire monitored space is.
- It calculates the `TOTAL_USABLE_AREA` by taking your defined room size (`WORLD_GRID_W * WORLD_GRID_H` or a `MANUAL_ROI` polygon).
- It subtracts the total area of all `STATIC_OBSTACLES`.
- `Room Density = Total People / TOTAL_USABLE_AREA (m²)`

## 4. Kernel Density Estimation (KDE) Heatmap
To visualize "hotspots" or pressure points smoothly on the UI, it doesn't just draw dots.
- It maps the physical locations onto a continuous grid.
- It applies a mathematically smoothed "Gaussian Blur" to each person. The radius of this blur is controlled by `KDE_BANDWIDTH` (representing the physical influence radius of a person in meters).
- It then forces the heat map to 0 over static obstacles, preventing "heat" from bleeding into physical objects like walls.

## Temporal Smoothing
To ensure the numbers don't rapidly jitter up and down every fraction of a second, the system uses an Exponential Moving Average (EMA) filter (`TemporalFilter`) and a rolling history buffer (`BUFFER_SIZE = 15` frames). This smooths out bounding box jitter and frame-to-frame density map fluctuations.
