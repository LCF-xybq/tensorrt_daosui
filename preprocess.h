/**
 * GPU Preprocessing for YOLO Detection
 *
 * Pipeline (all on GPU):
 * 1. BGR -> RGB
 * 2. Letterbox resize (preserve aspect ratio, gray 114 padding)
 * 3. Normalize / 255.0
 * 4. HWC -> CHW
 * 5. float32 -> half (FP16)
 *
 * Uses NPP for resize, custom CUDA kernels for the rest.
 * Assumes fixed input spatial size (1024x1024).
 */

#ifndef PREPROCESS_H_
#define PREPROCESS_H_

#include <vector>
#include <cuda_fp16.h>
#include <opencv2/opencv.hpp>

class GPUPreprocessor {
public:
    // targetW/targetH: model input spatial size (e.g., 1024x1024)
    GPUPreprocessor(int targetH, int targetW, int maxBatchSize);
    ~GPUPreprocessor();

    // Preprocess a batch of images on GPU.
    // Each img must be BGR uint8, same size.
    // Output: device buffer with FP16 CHW data [batchSize, 3, targetH, targetW].
    void preprocessBatch(const std::vector<cv::Mat>& imgs, void* outputDeviceFP16);

    // Get the device buffer for preprocessed output (FP16, pre-allocated for maxBatchSize)
    void* getDeviceBuffer() const { return deviceBufferFP16_; }
    void* getDeviceBufferFP32() const { return deviceRGBFloat_; }

    size_t getSingleImageBytes() const {
        return static_cast<size_t>(targetH_) * targetW_ * 3 * sizeof(__half);
    }

private:
    int targetH_;
    int targetW_;
    int maxBatchSize_;

    // Intermediate device buffers
    void* deviceBGRA_;        // Batch of BGR/RGB uint8 images (padded to 4 channels for NPP)
    void* deviceRGBFloat_;    // Batch of RGB float32 CHW
    void* deviceBufferFP16_;  // Final FP16 CHW output

    // NPP resize scratch buffer
    void* nppResizeScratch_;
    int nppResizeScratchBytes_;

    void allocateBuffers();
    void freeBuffers();
};

// Convert FP16 device buffer to FP32 device buffer (used when TensorRT outputs FP16)
void convertFP16ToFP32Device(const void* srcFP16, float* dstFP32,
                              int numElements, cudaStream_t stream);

#endif // PREPROCESS_H_
