# tensorrt_daosui

基于 TensorRT 的稻穗（rice panicle）检测推理项目。YOLOv10 模型在 GPU 上完成全流程（letterbox 预处理 → FP16 推理 → NMS 后处理），**支持任意尺寸输入图片**。

## 目录结构

```
tensorrt_daosui/
├── CMakeLists.txt                  # CMake 构建
├── pyproject.toml                  # Python 包配置
├── build_engine.py                 # ONNX 导出 + Engine 构建
├── pipeline.py                     # 完整推理流水线（替代 Deploy.py）
│
├── tensorrt_inference.h / .cc      # TensorRT 引擎加载与推理
├── preprocess.h / .cu              # GPU 预处理（letterbox resize + BGR→RGB + FP16）
├── yolo_postprocess.h / .cc        # YOLO 后处理 + NMS
├── pybind_module.cc                # pybind11 Python 绑定
│
├── python/tensorrt_daosui/
│   ├── __init__.py
│   └── postprocess.py              # Python 高层 API + letterbox 坐标逆映射
│
├── daosui_v1.pt                    # 训练好的 YOLOv10 权重
├── daosui_v1.engine                # TensorRT 序列化引擎
├── data/                           # 输入图片（支持任意尺寸）
└── results/                        # 输出结果
```

## 环境依赖

| 依赖 | 版本 | 路径 |
|------|------|------|
| CUDA Toolkit | 12.8 | `/usr/local/cuda` |
| TensorRT | 10.10.0.31 | `/home/lcf/Downloads/TensorRT-10.10.0.31` |
| pybind11 | 2.12+ | `/home/lcf/Downloads/pybind11` |
| OpenCV | 4.12.0 | `/usr/local` (编译安装) |
| Python | 3.12 | conda 环境 `deploy` |
| ultralytics | latest | 仅导出 ONNX 时需要 |

```bash
conda activate deploy
```

## 构建

整个构建分三步：导出 ONNX → 生成 Engine → 编译安装 Python 包。

### 1. 导出 ONNX

```bash
python build_engine.py --export --weights daosui_v1.pt
```

脚本会禁用 YOLOv10 的 end2end 模式，用 `torch.onnx.export` 直接导出原始格式 `[batch, 5, 8400]`（兼容 YOLOv8 后处理），在当前目录生成 `best.onnx`。

### 2. 生成 TensorRT Engine

```bash
python build_engine.py --build --onnx best.onnx --engine daosui_v1.engine
```

生成 `daosui_v1.engine`（FP16，动态 batch [1, 32]）。

也可以跳过脚本，直接用 trtexec：

```bash
trtexec --onnx=best.onnx --saveEngine=daosui_v1.engine --fp16 \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:32x3x640x640 \
    --maxShapes=images:32x3x640x640
```

### 3. 编译安装 Python 包

```bash
pip install -e . --no-build-isolation
```

验证安装：

```python
from tensorrt_daosui import WeedDetector, Detection
print("OK")
```

## 运行

### 完整流水线（推荐）

直接运行 `pipeline.py`，自动遍历 `data/` 下所有图片，批量推理，结果保存到 `results/`：

```bash
python pipeline.py
```

默认配置：

| 参数 | 值 | 说明 |
|------|----|------|
| `ENGINE_PATH` | `daosui_v1.engine` | Engine 文件 |
| `INPUT_DIR` | `data` | 输入图片目录 |
| `OUTPUT_DIR` | `results` | 输出目录 |
| `MODEL_SIZE` | 640 | 模型输入尺寸 |
| `SCORE_THRESHOLD` | 0.25 | 置信度阈值 |
| `NMS_THRESHOLD` | 0.5 | NMS IoU 阈值 |
| `MAX_BATCH_SIZE` | 32 | 批量大小 |

在 Python 中自定义参数：

```python
from pipeline import run_pipeline

run_pipeline(
    input_dir="./my_images",
    output_dir="./my_results",
    engine_path="./daosui_v1.engine",
)
```

### Python API 单独使用

```python
import cv2
from tensorrt_daosui import WeedDetector

# 初始化
det = WeedDetector("daosui_v1.engine", device_id=0)
det.load()

# 单张推理
img = cv2.imread("test.jpg")
results = det.detect_image(img)
for r in results:
    print(f"bbox={r.bbox}, score={r.score:.3f}, class={r.class_id}")

# 批量推理（支持不同尺寸图片混在一起）
imgs = [cv2.imread("a.png"), cv2.imread("b.jpg")]  # 512x512 + 1066x800
results = det.detect_batch(imgs)

det.unload()
```

返回的 `Detection` 对象：

```python
@dataclass
class Detection:
    bbox: list[float]   # [x1, y1, x2, y2] 原始图片坐标（非模型坐标）
    score: float        # 置信度
    class_id: int       # 类别 ID（0 = ricepanicle）
    batch_idx: int      # 批次中的图片索引
```

## 关于可变尺寸图片

输入图片可以是任意尺寸（如 512x512、1066x800 或其他）。处理流程：

1. **GPU Letterbox Resize**：将任意尺寸图片等比缩放到 640x640，空白区域填充灰色 (114,114,114)
2. **推理**：FP16 推理 + NMS 后处理，bbox 坐标在模型空间 (0~640)
3. **坐标逆映射**：Python 层自动将 bbox 从模型空间还原到原始图片坐标

整个过程对调用者透明，`detect_image` / `detect_batch` 返回的坐标始终是原图坐标。
