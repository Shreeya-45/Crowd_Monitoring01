"""Walkable Area Segmentation via BiSeNetV2 (Cityscapes).

Detects ground surfaces (road, sidewalk, optionally terrain) using a
pre-trained semantic segmentation model from mmsegmentation.  The module
is designed to degrade gracefully: if mmsegmentation is not installed or
model files are missing, it emits a warning and returns a full-frame mask
so that downstream pipeline stages continue to work.
"""

import warnings
from pathlib import Path

import cv2
import numpy as np


class GroundSegmentor:
    """Automatic walkable area detection using semantic segmentation.

    Uses BiSeNetV2 (Cityscapes-trained) to classify pixels as
    road / sidewalk / terrain.  Runs inference every *N*-th frame and
    caches the binary mask between runs to amortise the GPU cost.

    Falls back gracefully if ``mmsegmentation`` is not installed or if
    the model config / checkpoint files are missing.

    Typical usage::

        seg = GroundSegmentor(
            config_path='configs/bisenetv2.py',
            checkpoint_path='checkpoints/bisenetv2.pth',
            run_interval=30,
        )
        mask = seg.segment(frame)          # binary uint8, 1 = walkable
        roi_polygon = cv2.findContours(mask, ...)  # derive ROI polygon
    """

    # Cityscapes *train IDs* that correspond to walkable surfaces.
    # See: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    WALKABLE_CLASSES: dict[int, str] = {
        0: "road",
        1: "sidewalk",
        # 9: 'terrain' — optionally included via ``include_terrain``
    }

    def __init__(
        self,
        config_path: str = "",
        checkpoint_path: str = "",
        device: str = "cuda:0",
        include_terrain: bool = False,
        run_interval: int = 30,
    ):
        """Initialise the segmentor.

        Args:
            config_path:     Path to mmsegmentation model config (``.py``).
            checkpoint_path: Path to model checkpoint (``.pth``).
            device:          Torch device string (e.g. ``'cuda:0'``).
            include_terrain: Whether to treat Cityscapes class 9 (terrain)
                             as walkable.
            run_interval:    Run inference once every *N* frames; reuse the
                             cached mask in between.
        """
        self.config_path: str = config_path
        self.checkpoint_path: str = checkpoint_path
        self.device: str = device
        self.run_interval: int = run_interval
        self.include_terrain: bool = include_terrain

        self._model = None
        self._available: bool = False
        self._cached_mask: np.ndarray | None = None
        self._frame_count: int = 0

        # Build the set of Cityscapes train IDs we consider "walkable"
        self.walkable_ids: set[int] = set(self.WALKABLE_CLASSES.keys())
        if include_terrain:
            self.walkable_ids.add(9)

        # References to mmseg API functions (assigned on successful import)
        self._init_model_fn = None
        self._inference_model_fn = None

        self._try_load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _try_load_model(self) -> None:
        """Attempt to load the segmentation model.

        Emits a ``UserWarning`` and leaves ``self._available == False``
        when any prerequisite is missing so the rest of the pipeline
        can continue with manual ROI selection.
        """
        # --- Check that paths were supplied ---
        if not self.config_path or not self.checkpoint_path:
            warnings.warn(
                "GroundSegmentor: No model config/checkpoint specified. "
                "Auto-segmentation disabled. Use manual ROI instead."
            )
            return

        # --- Check that files exist on disk ---
        if not Path(self.config_path).exists() or not Path(self.checkpoint_path).exists():
            warnings.warn(
                f"GroundSegmentor: Model files not found. "
                f"Config: {self.config_path}, Checkpoint: {self.checkpoint_path}. "
                f"Auto-segmentation disabled."
            )
            return

        # --- Try importing mmsegmentation ---
        try:
            from mmseg.apis import inference_model, init_model  # type: ignore[import-untyped]

            self._init_model_fn = init_model
            self._inference_model_fn = inference_model

            self._model = init_model(
                self.config_path, self.checkpoint_path, device=self.device
            )
            self._available = True
            print(
                f"GroundSegmentor: BiSeNetV2 loaded successfully on {self.device}"
            )
        except ImportError:
            warnings.warn(
                "GroundSegmentor: mmsegmentation not installed. "
                "Install with: pip install -U openmim && "
                'mim install mmengine "mmcv>=2.0.0" mmsegmentation. '
                "Auto-segmentation disabled."
            )
        except Exception as exc:
            warnings.warn(f"GroundSegmentor: Failed to load model: {exc}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Whether the segmentation model was loaded successfully."""
        return self._available

    def segment(self, frame: np.ndarray, force: bool = False) -> np.ndarray:
        """Compute (or return cached) walkable-area mask.

        Inference is executed on the first call and then every
        ``run_interval`` frames.  Intermediate calls return the cached
        result.  If the frame resolution changes between calls the
        cached mask is resized with nearest-neighbour interpolation
        to preserve crisp boundaries.

        Args:
            frame: BGR image (``np.ndarray``).
            force: If ``True``, run segmentation immediately regardless
                   of the frame counter.

        Returns:
            Binary ``uint8`` mask of shape ``(H, W)`` where
            ``1`` = walkable and ``0`` = non-walkable.
            If the model is unavailable an all-ones mask is returned
            (i.e. the entire frame is treated as walkable).
        """
        h, w = frame.shape[:2]

        if not self._available:
            # Full-frame fallback when no model is loaded
            return np.ones((h, w), dtype=np.uint8)

        self._frame_count += 1

        # Reuse the cached mask unless it's time to re-segment
        if (
            not force
            and self._cached_mask is not None
            and self._frame_count % self.run_interval != 0
        ):
            # Handle resolution changes (e.g. dynamic window resize)
            if self._cached_mask.shape[:2] != (h, w):
                return cv2.resize(
                    self._cached_mask, (w, h), interpolation=cv2.INTER_NEAREST
                )
            return self._cached_mask

        # ---- Run inference ----
        result = self._inference_model_fn(self._model, frame)

        # ``pred_sem_seg.data`` is a torch Tensor of shape (1, H, W)
        seg_map: np.ndarray = result.pred_sem_seg.data.cpu().numpy().squeeze()

        # Build binary walkable mask from the Cityscapes train IDs
        mask = np.zeros((h, w), dtype=np.uint8)
        for cls_id in self.walkable_ids:
            mask[seg_map == cls_id] = 1

        # Morphological cleanup:
        #   CLOSE fills small holes inside walkable regions,
        #   OPEN removes isolated noisy blobs.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        self._cached_mask = mask
        return mask

    def get_cached_mask(self) -> np.ndarray | None:
        """Return the last computed walkable-area mask, or ``None``."""
        return self._cached_mask

    def reset(self) -> None:
        """Clear the cached mask and reset the frame counter."""
        self._cached_mask = None
        self._frame_count = 0
