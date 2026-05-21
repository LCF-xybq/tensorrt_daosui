# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TensorRT-accelerated rice panicle (稻穗) detection. A YOLOv10 model runs the full pipeline on GPU: letterbox preprocessing -> FP16 inference -> NMS postprocessing. Supports variable-size input images with automatic coordinate remapping.

## Build Commands

Conda environment: `conda activate deploy`

**Export ONNX** (disables YOLOv10 end2end to get raw `[batch, 5, 8400]` output):
```bash
python build_engine.py --export --weights daosui_v1.pt
```

**Build TensorRT engine** (FP16, dynamic batch [1, 32]):
```bash
python build_engine.py --build --onnx daosui_v1.onnx --engine daosui_v1.engine
```

**Compile and install Python package** (requires scikit-build-core):
```bash
pip install -e . --no-build-isolation
```

**Run inference pipeline:**
```bash
python pipeline.py
```

**Verify installation:**
```python
from tensorrt_daosui import WeedDetector, Detection
```

## Architecture

Two-layer design: C++ native module (via pybind11) handles GPU compute; Python wraps it with coordinate remapping.

### C++ Layer (compiled into `_tensorrt_daosui` .so)

- **`tensorrt_inference.h/.cc`** — Loads serialized `.engine` files via `nvinfer::IRuntime`, runs inference with `enqueueV3` (TensorRT 10.x API). Pre-allocates GPU buffers for max batch size. Uses `delete` (not `destroy()`) for TensorRT 10.x resource cleanup.

- **`preprocess.h/.cu`** — GPU preprocessing pipeline: H2D copy -> NPP letterbox resize (`nppiResizeSqrPixel_8u_C3R`, gray 114 padding) -> custom CUDA kernels for BGR->RGB + normalize + HWC->CHW (single kernel) -> float32->FP16 conversion. Links NPP sub-libraries: `nppc`, `nppig`, `nppial`.

- **`yolo_postprocess.h/.cc`** — Pure C++ YOLO postprocessing + NMS. Supports YOLOv5/v6/v7 (with objectness) and YOLOv8/v10/v11 (sigmoid, no objectness). No framework dependencies.

- **`pybind_module.cc`** — `WeedDetector` class exposed to Python. Releases GIL during preprocess+inference+postprocess. Returns `list[dict]` with `class_id`, `score`, `bbox` (x1,y1,x2,y2 in model space), `batch_idx`.

### Python Layer (`python/tensorrt_daosui/`)

- **`postprocess.py`** — `WeedDetector` high-level wrapper. Adds letterbox coordinate reversal so returned `Detection` objects have bbox in original image coordinates, not model 640x640 space. `Detection` dataclass: `bbox`, `score`, `class_id`, `batch_idx`.
- **`__init__.py`** — Re-exports `WeedDetector`, `WeedDetectorError`, `Detection`.

### Pipeline Scripts

- **`pipeline.py`** — End-to-end script: scans `data/` for images, batch-inferences, draws bounding boxes, saves to `results/`. Uses batch mode for throughput.
- **`build_engine.py`** — ONNX export (wraps YOLOv10 to disable end2end NMS) + `trtexec` engine conversion.
- **`Deploy.py`** — Legacy script, superseded by `pipeline.py`.

## Key Technical Details

- **Model input**: FP16 `[batch, 3, 640, 640]`, dynamic batch dim [1, 32]
- **Model output**: `[batch, 5, 8400]` where 5 = 4 bbox coords + 1 class (1-class YOLOv10)
- **Inference precision**: FP16 throughout (input and compute)
- **Letterbox**: GPU-side via NPP, gray padding (114,114,114), coordinate reversal in Python
- **TensorRT 10.x**: Uses `enqueueV3` not `enqueueV2`, `delete` not `destroy()`
- **CUDA 12.8 NPP**: Split into sub-libraries (`nppc`, `nppig`, `nppial`), no monolithic `libnppi`

## Dependencies & Paths

| Dependency | Version | Path |
|---|---|---|
| CUDA | 12.8 | `/usr/local/cuda` |
| TensorRT | 10.10.0.31 | `/home/lcf/Downloads/TensorRT-10.10.0.31` |
| pybind11 | 2.12+ | `/home/lcf/Downloads/pybind11` |
| OpenCV | 4.12.0 | `/usr/local` (compiled from source) |
| Python | 3.12 | conda env `deploy` |
| Build system | scikit-build-core + pybind11 + CMake (C++17/CUDA 17) | `pyproject.toml` + `CMakeLists.txt` |

## others
- When making a git commit, do not include "Co-Authored-By".
