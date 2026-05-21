/**
 * pybind11 Module: _tensorrt_daosui
 *
 * Exposes WeedDetector class with detect_single() and detect_batch().
 * Full pipeline: image -> GPU preprocess -> TensorRT inference -> YOLO postprocess -> detections.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include "tensorrt_inference.h"
#include "preprocess.h"
#include "yolo_postprocess.h"

namespace py = pybind11;

class WeedDetectorError : public std::exception {
public:
    WeedDetectorError(const std::string& msg) : message_(msg) {}
    const char* what() const noexcept override { return message_.c_str(); }
private:
    std::string message_;
};

class WeedDetector {
public:
    WeedDetector(const std::string& engine_path, int device_id,
                 int model_w, int model_h, int max_batch_size,
                 int yolo_version, int num_classes,
                 float score_threshold, float nms_threshold)
        : engine_path_(engine_path), device_id_(device_id),
          model_w_(model_w), model_h_(model_h),
          max_batch_size_(max_batch_size),
          yolo_version_(yolo_version), num_classes_(num_classes),
          score_threshold_(score_threshold), nms_threshold_(nms_threshold),
          inference_(nullptr), preprocessor_(nullptr),
          deviceOutput_(nullptr), is_init_(false) {}

    ~WeedDetector() { deinit(); }

    void init() {
        if (is_init_) return;

        cudaError_t ret = cudaSetDevice(device_id_);
        if (ret != cudaSuccess) {
            throw WeedDetectorError("cudaSetDevice failed: " + std::string(cudaGetErrorString(ret)));
        }

        inference_ = std::make_unique<TensorRTInference>(engine_path_, device_id_, max_batch_size_);
        if (!inference_->init()) {
            inference_.reset();
            throw WeedDetectorError("Failed to initialize TensorRT engine: " + engine_path_);
        }

        preprocessor_ = std::make_unique<GPUPreprocessor>(model_h_, model_w_, max_batch_size_);

        // Pre-allocate output buffer for max batch
        size_t maxOutputElements = static_cast<size_t>(max_batch_size_);
        auto outDims = inference_->getOutputDims();
        for (int d = 1; d < outDims.nbDims; ++d) {
            maxOutputElements *= outDims.d[d];
        }
        cudaMalloc(&deviceOutput_, maxOutputElements * sizeof(float));

        auto inDims = inference_->getInputDims();
        auto outDims2 = inference_->getOutputDims();

        std::ostringstream oss;
        oss << "Model loaded: " << engine_path_ << "\n";
        oss << "  Input dims: ";
        for (int d = 0; d < inDims.nbDims; ++d) {
            if (d > 0) oss << "x";
            oss << inDims.d[d];
        }
        oss << "\n  Output dims: ";
        for (int d = 0; d < outDims2.nbDims; ++d) {
            if (d > 0) oss << "x";
            oss << outDims2.d[d];
        }
        oss << "\n  Max batch: " << max_batch_size_;
        py::print(oss.str());

        is_init_ = true;
    }

    void deinit() {
        if (deviceOutput_) { cudaFree(deviceOutput_); deviceOutput_ = nullptr; }
        if (inference_) { inference_->deinit(); inference_.reset(); }
        preprocessor_.reset();
        is_init_ = false;
    }

    py::list detect_single(py::array_t<uint8_t, py::array::c_style> img_array) {
        if (!is_init_) {
            throw WeedDetectorError("WeedDetector not initialized. Call init() first.");
        }

        auto buf = img_array.request();
        if (buf.ndim != 3 || buf.shape[2] != 3) {
            throw WeedDetectorError("Input must be HxWx3 BGR uint8 image");
        }

        cv::Mat img(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);
        std::vector<cv::Mat> imgs = {img};

        return detectInternal(imgs, 1);
    }

    py::list detect_batch(py::list img_list) {
        if (!is_init_) {
            throw WeedDetectorError("WeedDetector not initialized. Call init() first.");
        }

        if (img_list.size() == 0) {
            return py::list();
        }

        if (static_cast<int>(img_list.size()) > max_batch_size_) {
            throw WeedDetectorError("Batch size " + std::to_string(img_list.size()) +
                                    " exceeds max_batch_size " + std::to_string(max_batch_size_));
        }

        std::vector<cv::Mat> imgs;
        imgs.reserve(img_list.size());

        for (auto item : img_list) {
            auto arr = item.cast<py::array_t<uint8_t, py::array::c_style>>();
            auto buf = arr.request();
            if (buf.ndim != 3 || buf.shape[2] != 3) {
                throw WeedDetectorError("Each image must be HxWx3 BGR uint8");
            }
            imgs.emplace_back(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);
        }

        return detectInternal(imgs, static_cast<int>(imgs.size()));
    }

    py::dict get_info() const {
        py::dict info;
        info["engine_path"] = engine_path_;
        info["device_id"] = device_id_;
        info["model_w"] = model_w_;
        info["model_h"] = model_h_;
        info["max_batch_size"] = max_batch_size_;
        info["yolo_version"] = yolo_version_;
        info["num_classes"] = num_classes_;
        return info;
    }

private:
    std::string engine_path_;
    int device_id_;
    int model_w_;
    int model_h_;
    int max_batch_size_;
    int yolo_version_;
    int num_classes_;
    float score_threshold_;
    float nms_threshold_;

    std::unique_ptr<TensorRTInference> inference_;
    std::unique_ptr<GPUPreprocessor> preprocessor_;
    void* deviceOutput_;
    bool is_init_;

    py::list detectInternal(const std::vector<cv::Mat>& imgs, int batchSize) {
        // GPU preprocess + inference (no GIL)
        std::vector<DetectBBoxResult> results;
        {
            py::gil_scoped_release release;

            void* inputBuffer = inference_->isInputFP16()
                ? preprocessor_->getDeviceBuffer()
                : preprocessor_->getDeviceBufferFP32();

            // Batch preprocess all images on GPU (supports variable sizes)
            preprocessor_->preprocessBatch(imgs, nullptr);

            // Run batch inference (all images in one enqueueV3 call)
            inference_->run(inputBuffer, deviceOutput_, batchSize);

            // Total output elements for all images
            auto outDims = inference_->getOutputDims();
            size_t singleOutputElements = 1;
            for (int d = 1; d < outDims.nbDims; ++d) {
                singleOutputElements *= outDims.d[d];
            }
            size_t totalOutputElements = static_cast<size_t>(batchSize) * singleOutputElements;

            std::vector<float> hostOutput(totalOutputElements);
            cudaMemcpy(hostOutput.data(), deviceOutput_,
                       totalOutputElements * sizeof(float), cudaMemcpyDeviceToHost);

            std::vector<int64_t> outputShape;
            outputShape.push_back(batchSize);
            for (int d = 1; d < outDims.nbDims; ++d) {
                outputShape.push_back(outDims.d[d]);
            }

            YoloPostProcess postprocess(yolo_version_, score_threshold_, nms_threshold_,
                                         num_classes_, model_h_, model_w_);
            postprocess.run(hostOutput, outputShape, results);
        }

        // Convert results to Python (with GIL)
        py::list output;
        for (const auto& det : results) {
            py::dict d;
            d["class_id"] = det.label_id_;
            d["score"] = det.score_;
            d["bbox"] = py::make_tuple(det.bbox_[0], det.bbox_[1], det.bbox_[2], det.bbox_[3]);
            d["batch_idx"] = det.index_;
            output.append(d);
        }
        return output;
    }
};

PYBIND11_MODULE(_tensorrt_daosui, m) {
    m.doc() = "TensorRT Rice Panicle Detection - pybind11 module";

    py::register_exception<WeedDetectorError>(m, "WeedDetectorError");

    py::class_<WeedDetector>(m, "WeedDetector")
        .def(py::init<const std::string&, int, int, int, int, int, int, float, float>(),
             py::arg("engine_path"),
             py::arg("device_id") = 0,
             py::arg("model_w") = 640,
             py::arg("model_h") = 640,
             py::arg("max_batch_size") = 32,
             py::arg("yolo_version") = 10,
             py::arg("num_classes") = 1,
             py::arg("score_threshold") = 0.25f,
             py::arg("nms_threshold") = 0.5f)
        .def("init", &WeedDetector::init,
             "Load engine and allocate GPU resources")
        .def("deinit", &WeedDetector::deinit,
             "Release engine and GPU resources")
        .def("detect_single", &WeedDetector::detect_single,
             py::arg("image"),
             "Detect in a single BGR uint8 image (HxWx3 numpy array)")
        .def("detect_batch", &WeedDetector::detect_batch,
             py::arg("images"),
             "Detect in a batch of BGR uint8 images (list of HxWx3 numpy arrays)")
        .def("get_info", &WeedDetector::get_info,
             "Get detector configuration info");
}
