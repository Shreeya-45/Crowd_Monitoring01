"""
Temporal Filter
===============
Exponential Moving Average (EMA) filter for density heatmaps.

Smooths frame-to-frame density map fluctuations to show trends.
    D_t = α · raw_t + (1 - α) · D_{t-1}

Adaptive alpha variant increases responsiveness when large changes
are detected (e.g., sudden crowd formation).
"""

import numpy as np


class TemporalFilter:
    """EMA smoothing filter for density heatmaps.

    Eliminates flicker from frame-to-frame detection jitter and shows
    density trends — areas that have been consistently crowded glow brighter.

    Args:
        alpha: Base smoothing factor (0-1). Higher = more responsive.
        adaptive: If True, alpha increases when large changes are detected.
    """

    def __init__(self, alpha: float = 0.3, adaptive: bool = True):
        self.base_alpha = alpha
        self.adaptive = adaptive
        self.smoothed = None  # Initialized on first frame

    def update(self, raw_heatmap: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing to a new raw heatmap.

        Args:
            raw_heatmap: Current frame's density map (float32).

        Returns:
            Smoothed density map (float32).
        """
        if self.smoothed is None or self.smoothed.shape != raw_heatmap.shape:
            # First frame or shape changed — initialize with raw values
            self.smoothed = raw_heatmap.copy()
            return self.smoothed

        if self.adaptive:
            # Higher alpha when large changes detected (faster response to
            # sudden crowd formation or dispersal)
            change = np.mean(np.abs(raw_heatmap - self.smoothed))
            alpha = np.clip(self.base_alpha + 2.0 * change, 0.1, 0.8)
        else:
            alpha = self.base_alpha

        self.smoothed = alpha * raw_heatmap + (1.0 - alpha) * self.smoothed
        return self.smoothed

    def reset(self):
        """Reset filter state (e.g., when ROI changes)."""
        self.smoothed = None
