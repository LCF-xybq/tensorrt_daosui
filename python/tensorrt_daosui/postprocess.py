"""TensorRT Rice Panicle Detection - Python API.

Wraps the C++ _tensorrt_daosui module with a Pythonic interface.
Handles letterbox coordinate reversal for variable-size input images.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from tensorrt_daosui._tensorrt_daosui import (
        DaoSuiWeedDetector as _WeedDetector,
        WeedDetectorError,
    )
except ImportError as e:
    raise ImportError(
        "Native module _tensorrt_daosui not found. "
        "Build with: pip install . (requires TensorRT, CUDA, and pybind11)"
    ) from e


__all__ = [
    "DaoSuiWeedDetector",
    "WeedDetectorError",
    "Detection",
]


@dataclass
class Detection:
    bbox: list[float]       # [x1, y1, x2, y2] in original image space
    score: float
    class_id: int
    batch_idx: int = 0


class DaoSuiWeedDetector:
    """High-level rice panicle detection API using TensorRT.

    Handles variable-size input images via letterbox resize.
    Detection bbox coordinates are in original image space (not model space).

    Usage::

        from tensorrt_daosui import DaoSuiWeedDetector

        det = DaoSuiWeedDetector("model.engine", device_id=0)
        det.load()

        results = det.detect_image(image)
        for r in results:
            print(r.bbox, r.score, r.class_id)

        det.unload()
    """

    def __init__(
        self,
        engine_path: str,
        device_id: int = 0,
        model_w: int = 640,
        model_h: int = 640,
        max_batch_size: int = 32,
        yolo_version: int = 10,
        num_classes: int = 1,
        score_threshold: float = 0.25,
        nms_threshold: float = 0.5,
    ):
        self._det = _WeedDetector(
            engine_path, device_id, model_w, model_h,
            max_batch_size, yolo_version, num_classes,
            score_threshold, nms_threshold,
        )
        self.model_w = model_w
        self.model_h = model_h
        self._loaded = False

    def load(self) -> None:
        self._det.init()
        self._loaded = True

    def unload(self) -> None:
        self._det.deinit()
        self._loaded = False

    def detect_image(self, image) -> list[Detection]:
        """Detect in a single BGR uint8 image (HxWx3 numpy array).

        Bbox coordinates are returned in original image space.
        """
        if not self._loaded:
            raise WeedDetectorError("Model not loaded. Call load() first.")
        orig_h, orig_w = image.shape[:2]
        raw = self._det.detect_single(image)
        return self._reverse_letterbox(raw, orig_w, orig_h)

    def detect_batch(self, images: list) -> list[Detection]:
        """Detect in a batch of BGR uint8 images (list of HxWx3 numpy arrays).

        Supports variable-size images within a batch.
        Bbox coordinates are returned in original image space.
        """
        if not self._loaded:
            raise WeedDetectorError("Model not loaded. Call load() first.")
        dims = [(img.shape[1], img.shape[0]) for img in images]  # (w, h)
        raw = self._det.detect_batch(images)
        return self._reverse_letterbox_batch(raw, dims)

    def _reverse_letterbox(
        self, raw_results, orig_w: int, orig_h: int,
    ) -> list[Detection]:
        """Reverse letterbox transform: model-space coords → original image coords."""
        scale = min(self.model_w / orig_w, self.model_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        pad_left = (self.model_w - new_w) / 2.0
        pad_top = (self.model_h - new_h) / 2.0

        dets = []
        for d in raw_results:
            x1, y1, x2, y2 = d["bbox"]
            x1 = max(0.0, (x1 - pad_left) / scale)
            y1 = max(0.0, (y1 - pad_top) / scale)
            x2 = min(float(orig_w), (x2 - pad_left) / scale)
            y2 = min(float(orig_h), (y2 - pad_top) / scale)
            dets.append(Detection(
                bbox=[x1, y1, x2, y2],
                score=float(d["score"]),
                class_id=int(d["class_id"]),
                batch_idx=int(d["batch_idx"]),
            ))
        return dets

    def _reverse_letterbox_batch(
        self, raw_results, dims: list[tuple[int, int]],
    ) -> list[Detection]:
        """Reverse letterbox for batch with per-image dimensions."""
        dets = []
        for d in raw_results:
            idx = int(d["batch_idx"])
            orig_w, orig_h = dims[idx]
            scale = min(self.model_w / orig_w, self.model_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            pad_left = (self.model_w - new_w) / 2.0
            pad_top = (self.model_h - new_h) / 2.0

            x1, y1, x2, y2 = d["bbox"]
            x1 = max(0.0, (x1 - pad_left) / scale)
            y1 = max(0.0, (y1 - pad_top) / scale)
            x2 = min(float(orig_w), (x2 - pad_left) / scale)
            y2 = min(float(orig_h), (y2 - pad_top) / scale)
            dets.append(Detection(
                bbox=[x1, y1, x2, y2],
                score=float(d["score"]),
                class_id=int(d["class_id"]),
                batch_idx=idx,
            ))
        return dets
