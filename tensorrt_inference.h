/**
 * TensorRT Inference Engine
 *
 * Loads a serialized TensorRT engine (.engine) and runs inference.
 * Pre-allocates GPU buffers for max batch size to avoid per-inference allocation.
 */

#ifndef TENSORRT_INFERENCE_H_
#define TENSORRT_INFERENCE_H_

#include <string>
#include <vector>
#include <memory>
#include <NvInfer.h>

struct CUctx_st;
typedef struct CUctx_st* CUcontext;

class TensorRTInference {
public:
    TensorRTInference(const std::string& enginePath, int deviceId,
                      int maxBatchSize = 32);
    ~TensorRTInference();

    bool init();
    void deinit();

    // Run inference with given batch size. Input/output must be pre-allocated GPU buffers.
    void run(const void* inputDevice, void* outputDevice, int batchSize);

    // Run inference for a single image. Copies one image's data from batch buffer.
    void runSingle(const void* inputDeviceBatch, void* outputDevice,
                   int imageIndex, int imageElements);

    int getMaxBatchSize() const { return maxBatchSize_; }
    size_t getInputSizeBytes() const;
    size_t getOutputSizeBytes() const;

    const nvinfer1::Dims& getInputDims() const { return inputDims_; }
    const nvinfer1::Dims& getOutputDims() const { return outputDims_; }
    bool isInputFP16() const { return inputIsFP16_; }
    bool isOutputFP16() const { return outputIsFP16_; }

private:
    std::string enginePath_;
    int deviceId_;
    int maxBatchSize_;

    CUcontext cudaContext_;
    void* cudaStream_;

    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;

    nvinfer1::Dims inputDims_;
    nvinfer1::Dims outputDims_;
    nvinfer1::DataType inputDataType_;
    nvinfer1::DataType outputDataType_;
    bool inputIsFP16_;
    bool outputIsFP16_;

    void* inputBinding_;
    void* outputBinding_;

    bool isInit_;
};

// Logger for TensorRT
class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override;
};

#endif // TENSORRT_INFERENCE_H_
