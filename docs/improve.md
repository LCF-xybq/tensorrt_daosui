# 性能对比：原生 PyTorch vs TensorRT

测试条件：相同 228 张图片，同一台机器。

## 总体对比

| 指标 | 原生 PyTorch | TensorRT (runSingle) | TensorRT (batch run) | 加速比 (batch vs PyTorch) |
|------|-------------|----------------------|----------------------|--------------------------|
| 模型/引擎加载 | 0.07s | 0.34s | 0.35s | — |
| 推理+IO 总耗时 | 23.53s | 2.93s | **2.72s** | **8.7x** |
| 单张平均（含IO） | ~103ms | ~12.8ms | **~11.9ms** | **8.7x** |
| 检测结果 | — | 2368 dets | 2368 dets | — |

## 优化历程

| 版本 | 推理+IO | 相对提升 | 改动 |
|------|---------|---------|------|
| 原生 PyTorch | 23.53s | 基准 | — |
| TensorRT + runSingle | 2.93s | 8.0x | GPU 预处理 + FP16 + TensorRT 引擎 |
| TensorRT + batch run | 2.72s | **8.7x** | `run(batch=32)` 替代逐张 `runSingle` 循环 |

batch run 相比 runSingle 节省了 32 次 `enqueueV3` 的 kernel launch 开销（0.21s / 228 张 ≈ 0.9ms/张）。

## 原生 PyTorch 各阶段耗时

| 阶段 | 单张耗时 | 说明 |
|------|---------|------|
| 读取图片 | ~10ms | cv2.imread |
| 推理 | 10~20ms (小图) / 1.5~2.7s (大图) | 首张 2.67s (CUDA 热身)，大图如 `DRPD_12m_0816`(2.67s)、`Different_Growth_Stages_0003`(1.49s) 明显更慢 |
| 后处理 | ~0ms | negligible |
| 绘制+保存 | ~10ms | cv2.imwrite |

PyTorch 逐张推理，无 batch。总耗时 23.53s 中，大图推理占了大头（仅 2 张大图就占了 ~4s）。

## TensorRT 各阶段耗时

| 阶段 | 耗时 | 说明 |
|------|------|------|
| 引擎加载 | 0.35s | 反序列化 .engine 文件 |
| 预处理+推理+后处理+IO | 2.72s | batch=32 一次 enqueueV3 |
| 每 batch 32 张 | ~370ms | 含预处理、推理、绘制、保存 |

TensorRT 预处理在 GPU 上完成（letterbox resize + BGR→RGB + normalize + FP16），后处理 NMS 在 CPU 上。

## 结论

TensorRT batch 方案相比原生 PyTorch 实现 **8.7 倍加速**，主要来自：
- FP16 半精度推理
- GPU 端预处理（避免 CPU→GPU 反复传输）
- batch 批量推理（一次 enqueueV3 处理 32 张，减少 kernel launch 开销）
