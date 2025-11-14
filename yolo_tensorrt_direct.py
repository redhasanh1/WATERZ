"""
Direct TensorRT YOLO Inference - MAXIMUM SPEED
Bypasses Ultralytics wrapper for extreme performance

Target: 1400+ fps (0.7ms per frame) with batch 64
"""

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Initializes CUDA context
import cv2
from typing import List, Tuple
import time


class DirectTensorRTYOLO:
    """
    Direct TensorRT YOLO inference engine
    - No Ultralytics overhead
    - Batch-optimized
    - GPU-only operations
    - Pre-allocated buffers
    """

    def __init__(self, engine_path: str, conf_thresh: float = 0.15, iou_thresh: float = 0.45):
        """
        Initialize direct TensorRT inference

        Args:
            engine_path: Path to TensorRT engine file
            conf_thresh: Confidence threshold for detections
            iou_thresh: IOU threshold for NMS
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # Load TensorRT engine
        print(f"[*] Loading TensorRT engine: {engine_path}")
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, 'rb') as f:
            engine_data = f.read()

        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        self.context = self.engine.create_execution_context()

        print(f"[+] Engine loaded!")
        print(f"   Batch size optimized: 64")

        # Get input/output bindings
        self.input_shape = None
        self.output_shape = None
        self.bindings = []
        self.binding_addrs = []

        for i in range(self.engine.num_bindings):
            binding_name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))

            if self.engine.binding_is_input(i):
                self.input_shape = shape
                print(f"   Input: {binding_name}, shape={shape}, dtype={dtype}")
            else:
                self.output_shape = shape
                print(f"   Output: {binding_name}, shape={shape}, dtype={dtype}")

        # Pre-allocate GPU buffers for maximum batch size
        self.max_batch_size = 128
        self.input_size = (640, 640)

        # Allocate CUDA memory
        self._allocate_buffers()

        print(f"[+] GPU buffers allocated (batch_size={self.max_batch_size})")
        print(f"[+] Ready for EXTREME SPEED inference!")

    def _allocate_buffers(self):
        """Pre-allocate CUDA buffers for input/output"""
        # Input buffer: [batch, 3, 640, 640] float32
        input_size = self.max_batch_size * 3 * 640 * 640 * np.dtype(np.float32).itemsize
        self.input_buffer = cuda.mem_alloc(input_size)

        # Output buffer: [batch, 8400, 85] float32 (for YOLOv8)
        # 8400 = 80x80 + 40x40 + 20x20 anchors
        # 85 = 4 (bbox) + 1 (obj conf) + 80 (classes)
        output_size = self.max_batch_size * 8400 * 85 * np.dtype(np.float32).itemsize
        self.output_buffer = cuda.mem_alloc(output_size)

    def preprocess_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess batch of images for YOLO

        Args:
            images: List of BGR images (H, W, 3)

        Returns:
            Preprocessed batch [B, 3, 640, 640] float32
        """
        batch_size = len(images)
        batch_input = np.zeros((batch_size, 3, 640, 640), dtype=np.float32)

        for i, img in enumerate(images):
            # Resize
            resized = cv2.resize(img, self.input_size, interpolation=cv2.INTER_LINEAR)

            # BGR to RGB
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Normalize to [0, 1]
            normalized = rgb.astype(np.float32) / 255.0

            # HWC to CHW
            transposed = normalized.transpose(2, 0, 1)

            batch_input[i] = transposed

        return batch_input

    def postprocess_batch(self, output: np.ndarray, batch_size: int) -> List[List[dict]]:
        """
        Postprocess YOLO output to bounding boxes

        Args:
            output: Raw YOLO output [B, 8400, 85]
            batch_size: Actual batch size

        Returns:
            List of detections per image
        """
        all_detections = []

        for b in range(batch_size):
            predictions = output[b]  # [8400, 85]

            # Extract boxes and scores
            boxes = predictions[:, :4]  # [8400, 4] - xywh
            obj_conf = predictions[:, 4]  # [8400] - objectness
            class_scores = predictions[:, 5:]  # [8400, 80]

            # Get class confidences
            class_conf = np.max(class_scores, axis=1)
            class_ids = np.argmax(class_scores, axis=1)

            # Final confidence
            confidences = obj_conf * class_conf

            # Filter by confidence
            mask = confidences > self.conf_thresh
            filtered_boxes = boxes[mask]
            filtered_conf = confidences[mask]
            filtered_classes = class_ids[mask]

            if len(filtered_boxes) == 0:
                all_detections.append([])
                continue

            # Convert xywh to xyxy
            xyxy_boxes = self._xywh_to_xyxy(filtered_boxes)

            # Apply NMS
            keep_indices = self._nms(xyxy_boxes, filtered_conf, self.iou_thresh)

            # Format results
            detections = []
            for idx in keep_indices:
                x1, y1, x2, y2 = xyxy_boxes[idx]
                detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': float(filtered_conf[idx]),
                    'class': int(filtered_classes[idx])
                })

            all_detections.append(detections)

        return all_detections

    def _xywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert YOLO xywh format to xyxy"""
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        return xyxy

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        Non-Maximum Suppression

        Args:
            boxes: [N, 4] xyxy format
            scores: [N] confidence scores
            iou_threshold: IOU threshold

        Returns:
            List of indices to keep
        """
        # Sort by score
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Compute IOU
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_order = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])

            iou = inter / (area_i + area_order - inter)

            # Keep boxes with IOU less than threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def detect_batch(self, images: List[np.ndarray], batch_size: int = 64) -> List[List[dict]]:
        """
        Run batch inference on images

        Args:
            images: List of BGR images
            batch_size: Batch size for inference (default 64 for optimal speed)

        Returns:
            List of detections per image
        """
        all_results = []
        num_images = len(images)

        # Process in batches
        for i in range(0, num_images, batch_size):
            batch_images = images[i:i + batch_size]
            actual_batch_size = len(batch_images)

            # Preprocess
            batch_input = self.preprocess_batch(batch_images)

            # Copy to GPU
            cuda.memcpy_htod(self.input_buffer, batch_input)

            # Set dynamic batch size
            self.context.set_binding_shape(0, (actual_batch_size, 3, 640, 640))

            # Run inference
            self.context.execute_v2([int(self.input_buffer), int(self.output_buffer)])

            # Copy result back to CPU
            output_size = actual_batch_size * 8400 * 85
            output = np.empty(output_size, dtype=np.float32)
            cuda.memcpy_dtoh(output, self.output_buffer)

            # Reshape
            output = output.reshape(actual_batch_size, 8400, 85)

            # Postprocess
            batch_results = self.postprocess_batch(output, actual_batch_size)
            all_results.extend(batch_results)

        return all_results

    def benchmark(self, num_frames: int = 300, batch_size: int = 64):
        """Benchmark inference speed"""
        print(f"\n[*] Benchmarking with {num_frames} frames, batch_size={batch_size}")

        # Create dummy images
        dummy_images = [np.zeros((640, 640, 3), dtype=np.uint8) for _ in range(num_frames)]

        # Warmup
        print("[*] Warmup...")
        self.detect_batch(dummy_images[:batch_size], batch_size=batch_size)

        # Benchmark
        print("[*] Running benchmark...")
        start = time.time()
        results = self.detect_batch(dummy_images, batch_size=batch_size)
        end = time.time()

        duration = end - start
        fps = num_frames / duration
        ms_per_frame = (duration / num_frames) * 1000

        print(f"\n[+] Benchmark Results:")
        print(f"   Total time: {duration:.3f}s")
        print(f"   FPS: {fps:.1f}")
        print(f"   ms per frame: {ms_per_frame:.2f}ms")
        print()

        return fps, ms_per_frame


# Test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python yolo_tensorrt_direct.py <engine_path>")
        print("Example: python yolo_tensorrt_direct.py runs/detect/new_sora_watermark/weights/best_int8_rtx5070.engine")
        sys.exit(1)

    engine_path = sys.argv[1]

    # Initialize
    detector = DirectTensorRTYOLO(engine_path)

    # Benchmark
    fps, ms_per_frame = detector.benchmark(num_frames=300, batch_size=64)

    if ms_per_frame < 1.0:
        print(f"🔥🔥🔥 EXTREME SPEED ACHIEVED! {ms_per_frame:.2f}ms per frame! 🔥🔥🔥")
    elif ms_per_frame < 2.0:
        print(f"🔥 EXCELLENT! {ms_per_frame:.2f}ms per frame!")
    else:
        print(f"⚡ Fast! {ms_per_frame:.2f}ms per frame")
