"""
稻穗检测 - TensorRT 完整处理流水线

替代 Deploy.py，使用 TensorRT 加速推理。
流程：
1. 加载 TensorRT engine
2. 遍历输入图片（支持可变尺寸）
3. GPU 预处理 + 推理 + 后处理 + letterbox 坐标逆映射
4. 绘制检测框 + 保存结果
5. 打印统计信息
"""

import os
import glob
import logging
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np

from tensorrt_daosui import WeedDetector
from tensorrt_daosui.postprocess import Detection

# 设为 1 时输出各步骤耗时
DEBUG_TIME = 1

logging.basicConfig(
    filename='processing.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    encoding='utf-8',
)
logger = logging.getLogger(__name__)


@contextmanager
def _timed(label: str):
    if DEBUG_TIME:
        t0 = time.perf_counter()
        yield
        logger.info(f"[计时] {label}: {time.perf_counter() - t0:.2f}s")
    else:
        yield

# ================== 配置 ==================
ENGINE_PATH = "daosui_v1.engine"
INPUT_DIR = "data"
OUTPUT_DIR = "results"

MODEL_SIZE = 640
MAX_BATCH_SIZE = 32
NUM_CLASSES = 1
YOLO_VERSION = 10
SCORE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.5
DEVICE_ID = 0

BOX_COLOR = (0, 0, 255)
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
# ==========================================


def get_image_paths(image_dir: str) -> list[str]:
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
    return sorted(paths)


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    result = image.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        cv2.rectangle(result, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        label = f"ricepanicle {det.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 1)
        cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw + 4, y1), BOX_COLOR, -1)
        cv2.putText(result, label, (x1 + 2, y1 - 4), FONT, 0.6, (255, 255, 255), 1)
    return result


def process_single_image(model: WeedDetector, image_path: str, save_path: str) -> int:
    img = cv2.imread(image_path)
    if img is None:
        logger.warning(f"Cannot read image: {image_path}")
        return 0

    try:
        detections = model.detect_image(img)
        num_dets = len(detections)
        logger.info(f"  {Path(image_path).name}: {num_dets} detections")

        img_with_boxes = draw_detections(img, detections)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img_with_boxes)
        return num_dets

    except Exception as e:
        logger.error(f"Failed to process {Path(image_path).name}: {e}")
        return 0


def process_batch(model: WeedDetector, image_paths: list[str], output_dir: str) -> int:
    """Process images in batches for higher throughput."""
    total_dets = 0
    success_count = 0

    for batch_start in range(0, len(image_paths), MAX_BATCH_SIZE):
        batch_paths = image_paths[batch_start:batch_start + MAX_BATCH_SIZE]

        batch_imgs = []
        valid_indices = []
        for i, path in enumerate(batch_paths):
            img = cv2.imread(path)
            if img is not None:
                batch_imgs.append(img)
                valid_indices.append(i)

        if not batch_imgs:
            continue

        batch_dets = model.detect_batch(batch_imgs)

        per_image: dict[int, list] = {}
        for det in batch_dets:
            per_image.setdefault(det.batch_idx, []).append(det)

        for local_idx, img in enumerate(batch_imgs):
            global_idx = batch_start + valid_indices[local_idx]
            img_name = Path(image_paths[global_idx]).name
            save_path = os.path.join(output_dir, img_name)

            dets = per_image.get(local_idx, [])
            img_with_boxes = draw_detections(img, dets)

            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(save_path, img_with_boxes)
            total_dets += len(dets)
            success_count += 1

        done = min(batch_start + MAX_BATCH_SIZE, len(image_paths))
        logger.info(f"Progress: [{done}/{len(image_paths)}]")

    logger.info(f"Batch processing complete: {success_count} images, {total_dets} detections")
    return total_dets


def run_pipeline(
    input_dir: str = INPUT_DIR,
    output_dir: str = OUTPUT_DIR,
    engine_path: str = ENGINE_PATH,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Rice Panicle Detection - TensorRT Pipeline")
    logger.info(f"Engine: {engine_path}")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Model size: {MODEL_SIZE}, Batch: {MAX_BATCH_SIZE}")
    logger.info(f"Thresholds: conf={SCORE_THRESHOLD}, nms={NMS_THRESHOLD}")
    logger.info("=" * 70)

    image_paths = get_image_paths(input_dir)
    if not image_paths:
        logger.error(f"No images found in {input_dir}")
        return

    logger.info(f"Found {len(image_paths)} images")

    with _timed("引擎加载"):
        model = WeedDetector(
            engine_path=engine_path,
            device_id=DEVICE_ID,
            model_w=MODEL_SIZE,
            model_h=MODEL_SIZE,
            max_batch_size=MAX_BATCH_SIZE,
            yolo_version=YOLO_VERSION,
            num_classes=NUM_CLASSES,
            score_threshold=SCORE_THRESHOLD,
            nms_threshold=NMS_THRESHOLD,
        )
        model.load()

    with _timed("推理(含IO)"):
        total_dets = process_batch(model, image_paths, output_dir)

    model.unload()

    logger.info("=" * 70)
    logger.info(f"Pipeline complete: {len(image_paths)} images, {total_dets} total detections")
    logger.info(f"Results saved to: {os.path.abspath(output_dir)}")
    logger.info("=" * 70)

    print(f"\nDone. {len(image_paths)} images processed, {total_dets} detections found.")
    print(f"Results: {os.path.abspath(output_dir)}")


if __name__ == '__main__':
    run_pipeline()
