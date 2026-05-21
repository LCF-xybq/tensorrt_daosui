/**
 * TensorRT Inference Implementation
 *
 * Workflow:
 * 1. cuCtxSetCurrent / cuCtxCreate - set CUDA context
 * 2. IRuntime::deserializeCudaEngine - load .engine file
 * 3. createExecutionContext - create inference context
 * 4. enqueueV3 - run inference with dynamic batch
 */

#include "tensorrt_inference.h"
#include "preprocess.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <fstream>
#include <iostream>
#include <cstring>
#include <cassert>

TRTLogger gLogger;

void TRTLogger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << "[TRT] " << msg << std::endl;
    }
}

TensorRTInference::TensorRTInference(const std::string& enginePath, int deviceId,
                                     int maxBatchSize)
    : enginePath_(enginePath),
      deviceId_(deviceId),
      maxBatchSize_(maxBatchSize),
      cudaContext_(nullptr),
      cudaStream_(nullptr),
      runtime_(nullptr),
      engine_(nullptr),
      context_(nullptr),
      inputBinding_(nullptr),
      outputBinding_(nullptr),
      inputIsFP16_(false),
      outputIsFP16_(false),
      isInit_(false) {}

TensorRTInference::~TensorRTInference() {
    if (isInit_) {
        deinit();
    }
}

bool TensorRTInference::init() {
    cudaError_t cuRet = cudaSetDevice(deviceId_);
    if (cuRet != cudaSuccess) {
        std::cerr << "cudaSetDevice failed: " << cudaGetErrorString(cuRet) << std::endl;
        return false;
    }

    cuRet = cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&cudaStream_));
    if (cuRet != cudaSuccess) {
        std::cerr << "cudaStreamCreate failed: " << cudaGetErrorString(cuRet) << std::endl;
        return false;
    }

    // Load serialized engine from file
    std::ifstream engineFile(enginePath_, std::ios::binary);
    if (!engineFile.is_open()) {
        std::cerr << "Failed to open engine file: " << enginePath_ << std::endl;
        return false;
    }

    engineFile.seekg(0, std::ios::end);
    size_t fileSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    std::vector<char> engineData(fileSize);
    engineFile.read(engineData.data(), fileSize);
    engineFile.close();

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    if (!runtime_) {
        std::cerr << "Failed to create TensorRT runtime" << std::endl;
        return false;
    }

    engine_ = runtime_->deserializeCudaEngine(engineData.data(), fileSize);
    if (!engine_) {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    // Query input/output dims (use profile 0 for shape)
    // The model has exactly 2 bindings: input and output
    assert(engine_->getNbIOTensors() == 2);

    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto dims = engine_->getTensorShape(name);
        auto mode = engine_->getTensorIOMode(name);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            inputDims_ = dims;
            inputDataType_ = engine_->getTensorDataType(name);
            inputIsFP16_ = (inputDataType_ == nvinfer1::DataType::kHALF);
            std::cout << "Input: name=" << name
                      << ", dims=" << dims.d[0];
            for (int d = 1; d < dims.nbDims; ++d)
                std::cout << "x" << dims.d[d];
            std::cout << ", dtype=" << (inputIsFP16_ ? "FP16" : "FP32")
                      << std::endl;
        } else {
            outputDims_ = dims;
            outputDataType_ = engine_->getTensorDataType(name);
            outputIsFP16_ = (outputDataType_ == nvinfer1::DataType::kHALF);
            std::cout << "Output: name=" << name
                      << ", dims=" << dims.d[0];
            for (int d = 1; d < dims.nbDims; ++d)
                std::cout << "x" << dims.d[d];
            std::cout << ", dtype=" << (outputIsFP16_ ? "FP16" : "FP32")
                      << std::endl;
        }
    }

    // Pre-allocate GPU buffers for max batch size
    size_t inputBytes = getInputSizeBytes();
    size_t outputBytes = static_cast<size_t>(maxBatchSize_);
    for (int d = 1; d < outputDims_.nbDims; ++d) {
        outputBytes *= outputDims_.d[d];
    }
    outputBytes *= outputIsFP16_ ? sizeof(__half) : sizeof(float);

    cuRet = cudaMalloc(&inputBinding_, inputBytes);
    if (cuRet != cudaSuccess) {
        std::cerr << "cudaMalloc input failed: " << cudaGetErrorString(cuRet) << std::endl;
        return false;
    }

    cuRet = cudaMalloc(&outputBinding_, outputBytes);
    if (cuRet != cudaSuccess) {
        std::cerr << "cudaMalloc output failed: " << cudaGetErrorString(cuRet) << std::endl;
        return false;
    }

    isInit_ = true;
    std::cout << "TensorRT Inference initialized successfully" << std::endl;
    return true;
}

void TensorRTInference::deinit() {
    if (inputBinding_) {
        cudaFree(inputBinding_);
        inputBinding_ = nullptr;
    }
    if (outputBinding_) {
        cudaFree(outputBinding_);
        outputBinding_ = nullptr;
    }
    if (context_) {
        delete context_;
        context_ = nullptr;
    }
    if (engine_) {
        delete engine_;
        engine_ = nullptr;
    }
    if (runtime_) {
        delete runtime_;
        runtime_ = nullptr;
    }
    if (cudaStream_) {
        cudaStreamDestroy(reinterpret_cast<cudaStream_t>(cudaStream_));
        cudaStream_ = nullptr;
    }
    isInit_ = false;
}

void TensorRTInference::run(const void* inputDevice, void* outputDevice, int batchSize) {
    if (!isInit_) {
        std::cerr << "TensorRT Inference not initialized" << std::endl;
        return;
    }

    size_t inputCopyBytes = static_cast<size_t>(batchSize);
    for (int d = 1; d < inputDims_.nbDims; ++d) {
        inputCopyBytes *= inputDims_.d[d];
    }
    inputCopyBytes *= inputIsFP16_ ? sizeof(__half) : sizeof(float);

    size_t outputElements = static_cast<size_t>(batchSize);
    for (int d = 1; d < outputDims_.nbDims; ++d) {
        outputElements *= outputDims_.d[d];
    }
    size_t outputCopyBytes = outputElements * (outputIsFP16_ ? sizeof(__half) : sizeof(float));

    cudaError_t cuRet = cudaMemcpyAsync(inputBinding_, inputDevice, inputCopyBytes,
                                         cudaMemcpyDeviceToDevice,
                                         reinterpret_cast<cudaStream_t>(cudaStream_));
    if (cuRet != cudaSuccess) {
        std::cerr << "cudaMemcpy input failed: " << cudaGetErrorString(cuRet) << std::endl;
        return;
    }

    const char* inputName = engine_->getIOTensorName(0);
    const char* outputName = engine_->getIOTensorName(1);

    // Set input shape for dynamic batch
    nvinfer1::Dims batchDims = inputDims_;
    batchDims.d[0] = batchSize;
    context_->setInputShape(inputName, batchDims);

    context_->setTensorAddress(inputName, inputBinding_);
    context_->setTensorAddress(outputName, outputBinding_);

    bool ok = context_->enqueueV3(reinterpret_cast<cudaStream_t>(cudaStream_));
    if (!ok) {
        std::cerr << "enqueueV3 failed" << std::endl;
        return;
    }

    if (outputIsFP16_) {
        convertFP16ToFP32Device(outputBinding_, static_cast<float*>(outputDevice),
                                static_cast<int>(outputElements),
                                reinterpret_cast<cudaStream_t>(cudaStream_));
    } else {
        cuRet = cudaMemcpyAsync(outputDevice, outputBinding_, outputCopyBytes,
                                 cudaMemcpyDeviceToDevice,
                                 reinterpret_cast<cudaStream_t>(cudaStream_));
        if (cuRet != cudaSuccess) {
            std::cerr << "cudaMemcpy output failed: " << cudaGetErrorString(cuRet) << std::endl;
            return;
        }
    }

    cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cudaStream_));
}

void TensorRTInference::runSingle(const void* inputDeviceBatch, void* outputDevice,
                                   int imageIndex, int imageElements) {
    if (!isInit_) {
        std::cerr << "TensorRT Inference not initialized" << std::endl;
        return;
    }

    auto stream = reinterpret_cast<cudaStream_t>(cudaStream_);
    const char* inputName = engine_->getIOTensorName(0);
    const char* outputName = engine_->getIOTensorName(1);

    // Single image size in bytes
    size_t singleInputBytes = static_cast<size_t>(imageElements)
                              * (inputIsFP16_ ? sizeof(__half) : sizeof(float));
    const char* src = static_cast<const char*>(inputDeviceBatch) + imageIndex * singleInputBytes;

    // Copy one image from batch buffer to engine input
    cudaMemcpyAsync(inputBinding_, src, singleInputBytes,
                    cudaMemcpyDeviceToDevice, stream);

    // Set input shape for dynamic batch (single image: batch=1)
    nvinfer1::Dims singleDims = inputDims_;
    singleDims.d[0] = 1;
    context_->setInputShape(inputName, singleDims);

    context_->setTensorAddress(inputName, inputBinding_);
    context_->setTensorAddress(outputName, outputBinding_);

    context_->enqueueV3(stream);

    // Output for one image
    size_t singleOutputElements = 1;
    for (int d = 1; d < outputDims_.nbDims; ++d) {
        singleOutputElements *= outputDims_.d[d];
    }
    size_t singleOutputBytes = singleOutputElements * (outputIsFP16_ ? sizeof(__half) : sizeof(float));
    char* dst = static_cast<char*>(outputDevice) + imageIndex * singleOutputElements * sizeof(float);

    if (outputIsFP16_) {
        convertFP16ToFP32Device(outputBinding_, reinterpret_cast<float*>(dst),
                                static_cast<int>(singleOutputElements), stream);
    } else {
        cudaMemcpyAsync(dst, outputBinding_, singleOutputBytes,
                        cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);
}

size_t TensorRTInference::getInputSizeBytes() const {
    size_t elements = static_cast<size_t>(maxBatchSize_);
    for (int d = 1; d < inputDims_.nbDims; ++d) {
        elements *= inputDims_.d[d];
    }
    return elements * (inputIsFP16_ ? sizeof(__half) : sizeof(float));
}

size_t TensorRTInference::getOutputSizeBytes() const {
    size_t elements = static_cast<size_t>(maxBatchSize_);
    for (int d = 1; d < outputDims_.nbDims; ++d) {
        elements *= outputDims_.d[d];
    }
    return elements * (outputIsFP16_ ? sizeof(__half) : sizeof(float));
}
