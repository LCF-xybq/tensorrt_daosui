# TensorRT 稻穗检测 - 实施方案

## 项目概述

用 TensorRT 框架 (C++)。保持和 `/home/lcf/Projects/ascendcl_daosui`（基于 AscendCL + pybind11 的 YOLO 推理项目），保持相同的项目结构。

### 源项目信息

- 模型: YOLO 检测模型 (`best.pt`)
- 输入: 待更新
- 输出: 待更新
- 1 个类别: 稻穗
- 全流程在 Deploy.py中，该文件在项目完成后不再需要
- 创建pipeline.py，作为全流程python脚本

### 环境依赖

| 依赖 | 路径/版本 |
|---|---|
| Conda 环境 | `deploy` (`conda activate deploy`) |
| CUDA | 12.8, `/usr/local/cuda` |
| TensorRT | 10.10.0.31, `/home/lcf/Downloads/TensorRT-10.10.0.31` |
| pybind11 | `/home/lcf/Downloads/pybind11` |
| OpenCV | 4.12.0, `/usr/local` (编译安装) |
| Python | 3.12 |

---

## 架构决策

| 决策项 | 选择 | 说明 |
|---|---|---|
| 模型转换 | trtexec 离线转换 | 避免 runtime builder 的启动延迟和额外依赖 |
| 推理精度 | FP16 | 输入和计算均使用 FP16 |
| Batch 策略 | 动态 batch `[1, 32]`, opt=32 | 适配单张和批量场景 |
| 空间尺寸 | 固定尺寸 | 不支持动态空间尺寸 |
| 预处理 | GPU (NPP + CUDA kernel) | NPP 做 resize, 自定义 kernel 做 BGR→RGB/normalize/HWC→CHW/FP16 转换 |
| 后处理 | 原封复用 `yolo_postprocess.cc` | 纯 C++ 标准库，无框架依赖 |
| C++ 封装 | 全管线: 图像→预处理→推理→后处理→检测结果 | Python 只做数据包装 |
| Python API | `detect_single(img)` + `detect_batch(imgs)` | C++ 返回原始数据, Python 侧包装为 Detection 数据类 |
| 内存管理 | 预分配最大 batch buffer | 避免每次推理 malloc/free |
| CUDA stream | 每个实例各自持有 | |
| 全局初始化 | 不需要 | TensorRT 不像 AscendCL 需要 aclInit/aclFinalize |

---

## 项目结构

```
tensorrt_daosui/
├── CMakeLists.txt              # 构建系统 (CUDA + TensorRT + NPP + pybind11)
├── pyproject.toml              # Python 包配置 (scikit-build-core)
├── tensorrt_inference.h        # TensorRT 推理引擎头文件
├── tensorrt_inference.cc       # TensorRT 推理引擎实现
├── preprocess.h                # GPU 预处理器头文件
├── preprocess.cu               # GPU 预处理 CUDA 实现 (NPP + 自定义 kernel)
├── yolo_postprocess.h          # YOLO 后处理头文件 (复用)
├── yolo_postprocess.cc         # YOLO 后处理实现 (复用, 已修复原始 bug)
├── pybind_module.cc            # pybind11 绑定 (_tensorrt_daosui 模块)
├── pipeline.py                 # 完整处理流水线 (重写)
├── daosui_v1.engine           # TensorRT 序列化引擎 (trtexec 生成)
├── data/                       # 输入数据目录
├── docs/                       # 文档
├── results/                    # 输出结果目录
└── python/
    └── tensorrt_daosui/
        ├── __init__.py          # 导出 WeedDetector, WeedDetectorError, Detection
        ├── postprocess.py       # Python 包装层 (Detection 数据类 + WeedDetector 高级 API)
        ├── distortion.py        # DJI 畸变校正 (复用)
        ├── slice.py             # 大图切片/拼接 (复用)
        └── stats.py             # 面积统计 (复用)
```

---

## 各模块详细设计

### 1. tensorrt_inference.h/cc — TensorRT 推理引擎

**职责**: 加载序列化的 `.engine` 文件, 管理推理生命周期, 执行推理。

**核心类**: `TensorRTInference`

```
构造(enginePath, deviceId, maxBatchSize=32)
  → init(): 反序列化 engine, 创建 execution context, 预分配 GPU buffer
  → run(inputDevice, outputDevice, batchSize): H2H copy → setInputShape → enqueueV3 → D2D copy
  → deinit(): 释放所有资源
```

**关键实现细节**:
- 使用 `nvinfer1::IRuntime::deserializeCudaEngine` 加载 engine
- 使用 `enqueueV3` (TensorRT 10.x API, 替代已废弃的 `enqueueV2`)
- 输入 buffer: FP16, `maxBatchSize × 3 × 1024 × 1024 × 2 bytes` = 最大 192MB
- 输出 buffer: FP32, `maxBatchSize × 7 × 21504 × 4 bytes` = 最大 ~18MB
- 每次 `run()` 先 `cudaMemcpyAsync` 从调用方 buffer 复制到预分配 buffer, 推理完再复制回去
- TensorRT 10.x 使用 `delete` 而非 `destroy()` 释放 engine/context/runtime

### 2. preprocess.h/cu — GPU 预处理

**职责**: 在 GPU 上完成图像预处理全流程。

**核心类**: `GPUPreprocessor`

**预处理管线** (全部在 GPU 上执行):
1. 输入图像 (BGR uint8) `cudaMemcpy` 到 GPU
2. 如果需要 resize: NPP `nppiResizeSqrPixel_8u_C3R` 做 letterbox resize (灰色 114 填充)
3. CUDA kernel: BGR→RGB + /255.0 归一化 + HWC→CHW (单 kernel)
4. CUDA kernel: float32→half (FP16) 转换

**CUDA Kernel 列表**:

| Kernel | 功能 | 线程映射 |
|---|---|---|
| `bgrToRgbNormalizeTransposeKernel` | BGR→RGB + 归一化 + HWC→CHW | 每线程处理一个像素 |
| `float32ToHalfKernel` | float32→FP16 转换 | 每线程处理一个元素 |
| `fillGrayKernel` | 灰色填充 (114,114,114) | 每线程处理一个像素 |

**NPP 库说明** (CUDA 12.8):
- 不再有单一的 `libnppi`, 已拆分为子库
- resize 函数位于 `libnppig` (geometry transforms)
- 链接需要: `nppc` (核心) + `nppig` (几何) + `nppial` (算术逻辑)
- API: `nppiResizeSqrPixel_8u_C3R` (参数为 xFactor/yFactor/xShift/yShift)

**内存分配**:
- `deviceBGRA_`: 最大 batch 的 uint8 BGR 图像
- `deviceRGBFloat_`: 最大 batch 的 float32 CHW 中间结果
- `deviceBufferFP16_`: 最大 batch 的 FP16 CHW 最终输出

### 3. yolo_postprocess.h/cc — YOLO 后处理

**职责**: 解析模型输出为检测结果, 执行 NMS。

**从源项目原封复用** (无 AscendCL 依赖), 修复了一个 bug:
- 原始 `runV8V11()` 第 206 行多了一个 `}`, 导致变量出作用域

**支持版本**: YOLOv5/v6/v7 (含 objectness) 和 YOLOv8/v11 (无 objectness, sigmoid 激活)

### 4. pybind_module.cc — Python 绑定

**职责**: 将 C++ 推理管线暴露给 Python。

**导出模块**: `_tensorrt_daosui`

**导出类**: `WeedDetector`

```python
WeedDetector(
    engine_path: str,
    device_id: int = 0,
    model_w: int = 1024,
    model_h: int = 1024,
    max_batch_size: int = 32,
    yolo_version: int = 8,
    num_classes: int = 3,
    score_threshold: float = 0.3,
    nms_threshold: float = 0.6,
)
```

**方法**:
- `init()` — 加载 engine, 创建 GPU preprocessor, 预分配输出 buffer
- `deinit()` — 释放所有资源
- `detect_single(image: ndarray[H,W,3] uint8) → list[dict]` — 单张推理
- `detect_batch(images: list[ndarray]) → list[dict]` — 批量推理
- `get_info() → dict` — 返回配置信息

**返回格式**: `list[dict]`, 每个 dict 包含:
- `class_id: int`
- `score: float`
- `bbox: tuple[x1, y1, x2, y2]`
- `batch_idx: int`

**GIL 管理**: 预处理+推理+后处理在 `py::gil_scoped_release` 中执行, 结果转换在 GIL 持有时执行。

### 5. Python 包 (python/tensorrt_daosui/)

**__init__.py**: 导出 `WeedDetector`, `WeedDetectorError`, `Detection`

**postprocess.py**:
- `Detection` 数据类: `bbox`, `score`, `class_id`
- `WeedDetector` 高级包装类: 封装 C++ `_WeedDetector`, 提供 `load()/unload()/detect_image()/detect_batch()`
- C++ 返回原始 dict, Python 包装为 `Detection` 对象

**distortion.py, slice.py, stats.py**: 从源项目原封复用。

### 6. pipeline.py — 完整流水线

**流程**:
1. DJI 畸变校正 (`DJIDistortionCorrector`)
2. 大图切片 (`slice_images`)
3. TensorRT GPU 推理 (`WeedDetector.detect_image` 逐张或 `detect_batch` 批量)
4. 切片还原拼接 (`DJIImageStitcher`)
5. 面积统计 (`calculate_real_areas`)
6. 清理归档 (`organize_and_cleanup`)

**与源项目的区别**:
- 移除了 `init_acl()/finalize_acl()` (TensorRT 不需要全局初始化)
- `WeedDetector` 构造参数更多 (engine_path, max_batch_size, yolo_version 等)
- `model.load()` 内部完成 engine 加载 + GPU 资源分配
- `detect_image()` 接收原始 BGR 图像, 内部完成全部预处理

### 7. CMakeLists.txt — 构建系统

**编译器**: GCC 11.4 (C++17) + NVCC 12.8 (CUDA 17)

**链接库**:
| 库 | 用途 |
|---|---|
| `nvinfer` | TensorRT 运行时 |
| `cudart` | CUDA 运行时 |
| `nppc` | NPP 核心 |
| `nppig` | NPP 几何变换 (resize) |
| `nppial` | NPP 算术逻辑 |
| `OpenCV` | 图像编解码 (imgcodecs, imgproc, core) |

**RPATH**: 编译时设置 TensorRT 和 CUDA 库路径, 运行时自动查找。

**构建工具链**: scikit-build-core + pybind11 + ninja

---

## 构建与安装

### 1. 生成动态 ONNX 模型

### 2. 转换为 TensorRT Engine

### 3. 构建 Python 包

```bash
conda activate deploy
pip install scikit-build-core build pybind11 numpy
python -m build --wheel
pip install dist/tensorrt_daosui-0.1.0-cp312-cp312-linux_x86_64.whl --force-reinstall
```

### 4. 验证安装

```python
from tensorrt_daosui import WeedDetector, WeedDetectorError, Detection
print("Import successful!")
```

---

## 使用示例

```python
import cv2
from tensorrt_daosui import WeedDetector, Detection

# 初始化
model = WeedDetector(
    engine_path="daosui_v1.engine",
    device_id=0,
    max_batch_size=32,
    yolo_version=8,
    num_classes=3,
    score_threshold=0.3,
    nms_threshold=0.6,
)
model.load()

# 单张推理
img = cv2.imread("test.jpg")
results = model.detect_image(img)
for det in results:
    print(f"Class: {det.class_id}, Score: {det.score:.2f}, BBox: {det.bbox}")

# 批量推理
imgs = [cv2.imread(f"img_{i}.jpg") for i in range(8)]
results = model.detect_batch(imgs)

# 释放
model.unload()
```

---

## 完整流水线运行

```bash
conda activate deploy
python pipeline.py
```

**pipeline.py 配置参数**:
- `ENGINE_PATH`: engine 文件路径
- `DEVICE_ID`: GPU 设备 ID
- `MODEL_SIZE`: 模型输入尺寸 (1024)
- `MAX_BATCH_SIZE`: 最大 batch (32)
- `NUM_CLASSES`: 检测类别数 (3)
- `SCORE_THRESHOLD`: 置信度阈值 (0.3)
- `NMS_THRESHOLD`: NMS IoU 阈值 (0.6)
- `GSD_M`: 地面采样距离 (0.007 米)
