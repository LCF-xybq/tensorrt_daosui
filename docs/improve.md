# 性能对比：原生 PyTorch vs TensorRT

测试条件：相同 228 张图片，同一台机器。

## 总体对比

| 指标 | 原生 PyTorch | TensorRT | 加速比 |
|------|-------------|----------|--------|
| 模型/引擎加载 | 0.07s | 0.34s | — |
| 推理+IO 总耗时 | 23.53s | 2.93s | **8.0x** |
| 单张平均（含IO） | ~103ms | ~12.8ms | **8.0x** |
| 检测结果 | — | 2368 dets | — |

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
| 引擎加载 | 0.34s | 反序列化 .engine 文件 |
| 预处理+推理+后处理+IO | 2.93s | 以 batch=32 批量处理 |
| 每 batch 32 张 | ~400ms | 含预处理、推理、绘制、保存 |

TensorRT 预处理在 GPU 上完成（letterbox resize + BGR→RGB + normalize + FP16），后处理 NMS 在 CPU 上。

## 已知瓶颈

当前 `pybind_module.cc` 中虽然按 batch 预处理图片，但推理是**逐张调用** `runSingle(batch=1)`，没有使用 TensorRT 的 batch 推理能力：

```cpp
// pybind_module.cc:188 — 当前实现
for (int i = 0; i < batchSize; ++i) {
    inference_->runSingle(inputBuffer, deviceOutput_, i, imageElements);
}
```

改为一次性 `run(batch=32)` 可减少 32 次 enqueueV3 调用的 kernel launch 开销，预计还能再提升推理速度。

## 结论

TensorRT 方案相比原生 PyTorch 已实现 **8 倍加速**，主要来自：
- FP16 半精度推理
- GPU 端预处理（避免 CPU→GPU 反复传输）
- batch 批量处理（预处理和 IO 部分）

进一步优化方向：将 `runSingle` 循环改为真正的 batch `run()`。
