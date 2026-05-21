/**
 * YOLO Postprocess Implementation - Detection Only
 *
 * V5/V6/V7 format: [batch, height, width, 5+num_classes]
 * V8/V11 format: [batch, (4+num_classes), anchors]
 */

#include "yolo_postprocess.h"
#include <cmath>
#include <algorithm>
#include <iostream>

YoloPostProcess::YoloPostProcess(int version, float scoreThreshold, float nmsThreshold,
                                 int numClasses, int modelH, int modelW)
    : version_(version),
      scoreThreshold_(scoreThreshold),
      nmsThreshold_(nmsThreshold),
      numClasses_(numClasses),
      modelH_(modelH),
      modelW_(modelW) {
}

YoloPostProcess::~YoloPostProcess() {
}

float YoloPostProcess::computeIOU(const std::array<float, 4>& box0,
                                  const std::array<float, 4>& box1) {
    const float area_i = (box0[3] - box0[1]) * (box0[2] - box0[0]);
    const float area_j = (box1[3] - box1[1]) * (box1[2] - box1[0]);
    if (area_i <= 0 || area_j <= 0) {
        return 0.0f;
    }

    const float intersection_x_min = std::max<float>(box0[0], box1[0]);
    const float intersection_y_min = std::max<float>(box0[1], box1[1]);
    const float intersection_x_max = std::min<float>(box0[2], box1[2]);
    const float intersection_y_max = std::min<float>(box0[3], box1[3]);

    const float intersection_area =
        std::max<float>(intersection_y_max - intersection_y_min, 0.0f) *
        std::max<float>(intersection_x_max - intersection_x_min, 0.0f);

    return intersection_area / (area_i + area_j - intersection_area);
}

void YoloPostProcess::computeNMS(const std::vector<DetectBBoxResult>& src,
                                 std::vector<int>& keepIdxs,
                                 float iouThreshold) {
    int n = src.size();

    std::vector<int> indices(n);
    for (int i = 0; i < n; ++i) indices[i] = i;

    std::sort(indices.begin(), indices.end(), [&src](int a, int b) {
        return src[a].score_ > src[b].score_;
    });

    keepIdxs.clear();
    std::vector<bool> suppressed(n, false);

    for (int i = 0; i < n; ++i) {
        int idx = indices[i];
        if (suppressed[idx]) continue;

        keepIdxs.push_back(idx);

        for (int j = i + 1; j < n; ++j) {
            int idx_j = indices[j];
            if (suppressed[idx_j]) continue;

            float iou = computeIOU(src[idx].bbox_, src[idx_j].bbox_);
            if (iou > iouThreshold) {
                suppressed[idx_j] = true;
            }
        }
    }
}

void YoloPostProcess::run(const std::vector<float>& outputData,
                          const std::vector<int64_t>& outputShape,
                          std::vector<DetectBBoxResult>& results) {
    const float* data = outputData.data();

    if (version_ == 5 || version_ == 6 || version_ == 7) {
        runV5V6(data, outputShape, results);
    } else {
        runV8V11(data, outputShape, results);
    }
}

void YoloPostProcess::runV5V6(const float* data, const std::vector<int64_t>& shape,
                              std::vector<DetectBBoxResult>& results) {
    int batch = shape[0];
    int height = shape[1];
    int width = shape[2];
    int stride = 5 + numClasses_;

    for (int b = 0; b < batch; ++b) {
        const float* dataBatch = data + b * height * width * stride;

        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                const float* dataRow = dataBatch + (h * width + w) * stride;

                float x_center = dataRow[0];
                float y_center = dataRow[1];
                float box_w = dataRow[2];
                float box_h = dataRow[3];
                float objectness = dataRow[4];

                float x0 = x_center - box_w * 0.5f;
                float y0 = y_center - box_h * 0.5f;
                float x1 = x_center + box_w * 0.5f;
                float y1 = y_center + box_h * 0.5f;

                x0 = std::max(0.0f, x0);
                y0 = std::max(0.0f, y0);
                x1 = std::min(static_cast<float>(modelW_), x1);
                y1 = std::min(static_cast<float>(modelH_), y1);

                for (int c = 0; c < numClasses_; ++c) {
                    float score = objectness * dataRow[5 + c];
                    if (score > scoreThreshold_) {
                        DetectBBoxResult bbox;
                        bbox.index_ = b;
                        bbox.label_id_ = c;
                        bbox.score_ = score;
                        bbox.bbox_[0] = x0 / modelW_;
                        bbox.bbox_[1] = y0 / modelH_;
                        bbox.bbox_[2] = x1 / modelW_;
                        bbox.bbox_[3] = y1 / modelH_;
                        results.push_back(bbox);
                    }
                }
            }
        }
    }

    std::vector<int> keepIdxs;
    computeNMS(results, keepIdxs, nmsThreshold_);

    std::vector<DetectBBoxResult> filtered;
    for (int idx : keepIdxs) {
        filtered.push_back(results[idx]);
    }
    results = filtered;
}

void YoloPostProcess::runV8V11(const float* data, const std::vector<int64_t>& shape,
                              std::vector<DetectBBoxResult>& results) {
    int batch = 1;
    int numAnchors = 0;
    int featuresPerAnchor = 0;

    if (shape.size() == 3) {
        batch = shape[0];
        featuresPerAnchor = shape[1];
        numAnchors = shape[2];
    } else if (shape.size() == 2) {
        featuresPerAnchor = shape[0];
        numAnchors = shape[1];
    } else {
        batch = shape[0];
        for (size_t i = 1; i < shape.size(); ++i) {
            numAnchors *= shape[i];
        }
    }

    int stride = featuresPerAnchor * numAnchors;

    for (int b = 0; b < batch; ++b) {
        const float* batchData = data + b * stride;

        std::vector<DetectBBoxResult> batchResults;

        for (int anchor = 0; anchor < numAnchors; ++anchor) {
            float cx = batchData[0 * numAnchors + anchor];
            float cy = batchData[1 * numAnchors + anchor];
            float w = batchData[2 * numAnchors + anchor];
            float h = batchData[3 * numAnchors + anchor];

            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;
            float x2 = cx + w * 0.5f;
            float y2 = cy + h * 0.5f;

            if (x1 < 0 || x1 > modelW_ || y1 < 0 || y1 > modelH_ ||
                x2 < 0 || x2 > modelW_ || y2 < 0 || y2 > modelH_) {
                continue;
            }
            if (x2 < x1 || y2 < y1) {
                continue;
            }

            int bestClass = 0;
            float bestScore = 0.0f;

            int actualNumClasses = std::min(featuresPerAnchor - 4, numClasses_);
            for (int c = 0; c < actualNumClasses; ++c) {
                float score = batchData[(4 + c) * numAnchors + anchor];
                if (score > bestScore) {
                    bestScore = score;
                    bestClass = c;
                }
            }

            if (bestScore > scoreThreshold_) {
                DetectBBoxResult bbox;
                bbox.index_ = b;
                bbox.label_id_ = bestClass;
                bbox.score_ = bestScore;
                bbox.bbox_[0] = x1;
                bbox.bbox_[1] = y1;
                bbox.bbox_[2] = x2;
                bbox.bbox_[3] = y2;
                batchResults.push_back(bbox);
            }
        }

        // Per-image NMS
        std::vector<int> keepIdxs;
        computeNMS(batchResults, keepIdxs, nmsThreshold_);
        for (int idx : keepIdxs) {
            results.push_back(batchResults[idx]);
        }
    }
}
