/**
 * YOLO Postprocess - Detection Only (no keypoints)
 *
 * Supports YOLOv5/v6/v7/v8/v11 detection models.
 */

#ifndef YOLO_POSTPROCESS_H_
#define YOLO_POSTPROCESS_H_

#include <vector>
#include <string>
#include <array>

struct DetectBBoxResult {
    int index_;          // Batch index
    int label_id_;       // Class ID
    float score_;        // Confidence score
    std::array<float, 4> bbox_;  // [xmin, ymin, xmax, ymax]
};

class YoloPostProcess {
public:
    YoloPostProcess(int version, float scoreThreshold, float nmsThreshold,
                   int numClasses, int modelH, int modelW);
    ~YoloPostProcess();

    void run(const std::vector<float>& outputData,
             const std::vector<int64_t>& outputShape,
             std::vector<DetectBBoxResult>& results);

    void runV5V6(const float* data, const std::vector<int64_t>& shape,
                 std::vector<DetectBBoxResult>& results);

    void runV8V11(const float* data, const std::vector<int64_t>& shape,
                  std::vector<DetectBBoxResult>& results);

    static float computeIOU(const std::array<float, 4>& box0,
                            const std::array<float, 4>& box1);

    static void computeNMS(const std::vector<DetectBBoxResult>& src,
                           std::vector<int>& keepIdxs,
                           float iouThreshold);

private:
    int version_;
    float scoreThreshold_;
    float nmsThreshold_;
    int numClasses_;
    int modelH_;
    int modelW_;
};

#endif // YOLO_POSTPROCESS_H_
