/**
 * GPU Preprocessing CUDA Implementation
 *
 * Uses NPP for resize (nppiResizeSqrPixel), custom CUDA kernels for
 * BGR->RGB, normalize, HWC->CHW, and float32->half conversion.
 *
 * Supports variable-size input images within a batch.
 */

#include "preprocess.h"
#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <cuda_fp16.h>
#include <iostream>
#include <cstring>
#include <algorithm>

// ============================================================
// CUDA Kernels
// ============================================================

// Kernel: BGR uint8 -> RGB float32, normalize by /255, transpose HWC -> CHW
// Input: batch of BGR images in HWC layout [N, H, W, 3] (uint8)
// Output: float32 CHW [N, 3, H, W]
__global__ void bgrToRgbNormalizeTransposeKernel(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int height, int width, int batchSize)
{
    int totalPixels = height * width * batchSize;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= totalPixels) return;

    int n = idx / (height * width);
    int hw = idx % (height * width);
    int h = hw / width;
    int w = hw % width;

    int srcIdx = idx * 3; // HWC layout

    unsigned char b = src[srcIdx + 0];
    unsigned char g = src[srcIdx + 1];
    unsigned char r = src[srcIdx + 2];

    float rf = static_cast<float>(r) / 255.0f;
    float gf = static_cast<float>(g) / 255.0f;
    float bf = static_cast<float>(b) / 255.0f;

    int planeSize = height * width;

    dst[n * 3 * planeSize + 0 * planeSize + h * width + w] = rf;
    dst[n * 3 * planeSize + 1 * planeSize + h * width + w] = gf;
    dst[n * 3 * planeSize + 2 * planeSize + h * width + w] = bf;
}

// Kernel: float32 -> half conversion
__global__ void float32ToHalfKernel(
    const float* __restrict__ src,
    half* __restrict__ dst,
    int totalElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;
    dst[idx] = __float2half(src[idx]);
}

// Kernel: half -> float32 conversion
__global__ void halfToFloatKernel(
    const half* __restrict__ src,
    float* __restrict__ dst,
    int totalElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;
    dst[idx] = __half2float(src[idx]);
}

void convertFP16ToFP32Device(const void* srcFP16, float* dstFP32,
                              int numElements, cudaStream_t stream)
{
    int threads = 256;
    int blocks = (numElements + threads - 1) / threads;
    halfToFloatKernel<<<blocks, threads, 0, stream>>>(
        static_cast<const half*>(srcFP16), dstFP32, numElements);
}

// Kernel: fill uint8 BGR image with gray (114, 114, 114)
__global__ void fillGrayKernel(
    unsigned char* __restrict__ dst, int totalPixels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalPixels) return;
    dst[idx * 3 + 0] = 114;
    dst[idx * 3 + 1] = 114;
    dst[idx * 3 + 2] = 114;
}

// ============================================================
// GPUPreprocessor Implementation
// ============================================================

GPUPreprocessor::GPUPreprocessor(int targetH, int targetW, int maxBatchSize)
    : targetH_(targetH), targetW_(targetW), maxBatchSize_(maxBatchSize),
      deviceBGRA_(nullptr), deviceRGBFloat_(nullptr), deviceBufferFP16_(nullptr),
      nppResizeScratch_(nullptr), nppResizeScratchBytes_(0) {
    allocateBuffers();
}

GPUPreprocessor::~GPUPreprocessor() {
    freeBuffers();
}

void GPUPreprocessor::allocateBuffers() {
    size_t singleImageBytes = static_cast<size_t>(targetH_) * targetW_ * 3;

    cudaMalloc(&deviceBGRA_, maxBatchSize_ * singleImageBytes);
    cudaMalloc(&deviceRGBFloat_, maxBatchSize_ * targetH_ * targetW_ * 3 * sizeof(float));
    cudaMalloc(&deviceBufferFP16_, maxBatchSize_ * targetH_ * targetW_ * 3 * sizeof(half));
}

void GPUPreprocessor::freeBuffers() {
    if (deviceBGRA_) { cudaFree(deviceBGRA_); deviceBGRA_ = nullptr; }
    if (deviceRGBFloat_) { cudaFree(deviceRGBFloat_); deviceRGBFloat_ = nullptr; }
    if (deviceBufferFP16_) { cudaFree(deviceBufferFP16_); deviceBufferFP16_ = nullptr; }
    if (nppResizeScratch_) { cudaFree(nppResizeScratch_); nppResizeScratch_ = nullptr; }
}

void GPUPreprocessor::preprocessBatch(const std::vector<cv::Mat>& imgs, void* outputDeviceFP16) {
    int batchSize = static_cast<int>(imgs.size());

    size_t targetSingleImageBGRBytes = static_cast<size_t>(targetH_) * targetW_ * 3;

    // Per-image upload and letterbox resize (supports variable input sizes)
    for (int i = 0; i < batchSize; ++i) {
        int srcH_i = imgs[i].rows;
        int srcW_i = imgs[i].cols;
        size_t srcBytes_i = static_cast<size_t>(srcH_i) * srcW_i * 3;

        unsigned char* dstBase = static_cast<unsigned char*>(deviceBGRA_) +
                                 i * targetSingleImageBGRBytes;

        if (srcH_i == targetH_ && srcW_i == targetW_) {
            // No resize needed - upload directly
            cudaMemcpy(dstBase, imgs[i].data, srcBytes_i, cudaMemcpyHostToDevice);
        } else {
            // Letterbox resize: preserve aspect ratio, gray (114) padding
            double scale = std::min(static_cast<double>(targetW_) / srcW_i,
                                    static_cast<double>(targetH_) / srcH_i);
            int newW = static_cast<int>(srcW_i * scale);
            int newH = static_cast<int>(srcH_i * scale);
            int top = (targetH_ - newH) / 2;
            int left = (targetW_ - newW) / 2;

            // Fill target with gray padding
            int totalPixels = targetH_ * targetW_;
            int threads = 256;
            int blocks = (totalPixels + threads - 1) / threads;
            fillGrayKernel<<<blocks, threads>>>(dstBase, totalPixels);

            // Upload source image to temp buffer (reuse deviceRGBFloat_)
            unsigned char* tempSrc = static_cast<unsigned char*>(deviceRGBFloat_);
            cudaMemcpy(tempSrc, imgs[i].data, srcBytes_i, cudaMemcpyHostToDevice);

            // NPP resize into the correct region
            NppiSize srcSize = {srcW_i, srcH_i};
            NppiRect srcROI = {0, 0, srcW_i, srcH_i};
            NppiRect dstROI = {left, top, newW, newH};

            double resizeXFactor = static_cast<double>(targetW_) / srcW_i;
            double resizeYFactor = static_cast<double>(targetH_) / srcH_i;

            NppStatus status = nppiResizeSqrPixel_8u_C3R(
                tempSrc, srcSize, srcW_i * 3, srcROI,
                dstBase, targetW_ * 3, dstROI,
                resizeXFactor, resizeYFactor, 0.0, 0.0,
                NPPI_INTER_LINEAR);

            if (status != NPP_NO_ERROR) {
                std::cerr << "NPP resize failed for image " << i
                          << " (src=" << srcW_i << "x" << srcH_i
                          << ") with status: " << status << std::endl;
                return;
            }
        }
    }

    // BGR->RGB + normalize + HWC->CHW (operates on target-size data, same for all)
    {
        int totalPixels = batchSize * targetH_ * targetW_;
        int threads = 256;
        int blocks = (totalPixels + threads - 1) / threads;

        bgrToRgbNormalizeTransposeKernel<<<blocks, threads>>>(
            static_cast<const unsigned char*>(deviceBGRA_),
            static_cast<float*>(deviceRGBFloat_),
            targetH_, targetW_, batchSize);
    }

    // float32 -> half (FP16)
    {
        int totalElements = batchSize * 3 * targetH_ * targetW_;
        int threads = 256;
        int blocks = (totalElements + threads - 1) / threads;

        float32ToHalfKernel<<<blocks, threads>>>(
            static_cast<const float*>(deviceRGBFloat_),
            static_cast<half*>(outputDeviceFP16 ? outputDeviceFP16 : deviceBufferFP16_),
            totalElements);
    }

    cudaDeviceSynchronize();
}
