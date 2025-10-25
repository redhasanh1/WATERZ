import sys
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
import os
try:
    import torch
    # PyTorch 2.6+ compatibility: Globally disable weights_only restriction
    # This allows loading custom-trained YOLO models
    if hasattr(torch, 'load'):
        _original_torch_load = torch.load
        def _patched_load(*args, **kwargs):
            # Force weights_only=False for all torch.load calls
            kwargs.setdefault('weights_only', False)
            return _original_torch_load(*args, **kwargs)
        torch.load = _patched_load
except Exception:
    torch = None

class YOLOWatermarkDetector:
    def __init__(self, model_path=None, require_tensorrt=None):
        """
        Initialize YOLOv8 watermark detector - BATCH TENSORRT ONLY!

        Args:
            model_path: Path to custom YOLOv8 model, or None to use batch TensorRT engine
            require_tensorrt: Ignored - always uses batch TensorRT engine (no fallbacks)
        """
        try:
            print("Loading YOLOv8 watermark detection model...")
            from ultralytics import YOLO

            # ALWAYS require batch TensorRT - no fallbacks!
            self._require_tensorrt = True
            self._using_tensorrt = False

            if model_path:
                # Use custom model
                self.model = YOLO(model_path)
                print(f"✅ Loaded custom model: {model_path}")
            else:
                # 🔥 BATCH-ENABLED ENGINE ONLY - NO FALLBACKS!
                batch_engine = 'runs/detect/new_sora_watermark/weights/best_fp16_batch_rtx5070.engine'

                if not os.path.exists(batch_engine):
                    raise RuntimeError(
                        f"❌ BATCH ENGINE NOT FOUND: {batch_engine}\n"
                        f"   Run BUILD_FP16_BATCH.bat to create it!\n"
                        f"   NO FALLBACKS ALLOWED - EXTREME SPEED MODE ONLY!"
                    )

                # Check CUDA available
                if (torch is None) or (not torch.cuda.is_available()):
                    raise RuntimeError(
                        "❌ CUDA GPU not available!\n"
                        f"   Batch TensorRT requires GPU\n"
                        f"   Check that PyTorch CUDA is installed and GPU is accessible"
                    )

                # Load batch-enabled TensorRT engine
                print(f"🔥 Loading BATCH-ENABLED TensorRT engine: {batch_engine}")
                try:
                    self.model = YOLO(batch_engine, task='detect')
                    print(f"✅ Batch TensorRT engine loaded! (10.7 MB, batch 1-128)")
                    print(f"   Expected performance: 1-2ms per frame (600-1000 fps)")
                    self._using_tensorrt = True
                except Exception as e:
                    raise RuntimeError(
                        f"❌ Failed to load batch TensorRT engine: {e}\n"
                        f"   Engine: {batch_engine}\n"
                        f"   NO FALLBACKS - Fix the engine or rebuild it!"
                    ) from e

            self.use_yolo = True
            print("✅ YOLOv8 ready for watermark detection!")

        except Exception as e:
            print(f"❌ Could not load YOLOv8 batch TensorRT engine: {e}")
            # NO FALLBACKS - fail fast!
            raise RuntimeError(
                f"❌ BATCH TENSORRT ENGINE REQUIRED!\n"
                f"   Error: {e}\n"
                f"   Run BUILD_FP16_BATCH.bat to create the engine\n"
                f"   NO FALLBACKS ALLOWED - EXTREME SPEED MODE ONLY!"
            ) from e

    def detect(self, image, confidence_threshold=0.25, padding=30):
        """
        Detect watermarks in image

        Args:
            image: numpy array (H, W, 3) BGR
            confidence_threshold: Minimum confidence for detections (0-1)
            padding: Pixels to add around detected region

        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...] or empty list
        """
        if not self.use_yolo:
            return []

        h, w = image.shape[:2]

        # Run YOLOv8 detection
        # Force GPU when TensorRT mode is required (no CPU fallback!)
        if self._require_tensorrt:
            # TensorRT mode - MUST use GPU, fail if unavailable
            if torch is None or not torch.cuda.is_available():
                raise RuntimeError(
                    "TensorRT mode requires CUDA GPU but torch.cuda.is_available() is False. "
                    "Check that PyTorch CUDA is installed and GPU is accessible."
                )
            device_arg = 0  # GPU only
        else:
            # Regular mode - allow CPU fallback
            device_arg = 0 if (torch is not None and torch.cuda.is_available()) else 'cpu'

        results = self.model(image, conf=confidence_threshold, device=device_arg, verbose=False)

        bboxes = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Add padding
                x1 = max(0, int(x1) - padding)
                y1 = max(0, int(y1) - padding)
                x2 = min(w, int(x2) + padding)
                y2 = min(h, int(y2) + padding)

                conf = float(box.conf[0])

                bboxes.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf
                })

                # Removed confidence logging - progress shown via task updates instead
                pass

        return bboxes

    def detect_batch(self, images, confidence_threshold=0.25, padding=30, batch_size=64):
        """
        Batch detect watermarks in multiple images (EXTREME SPEED!)

        Args:
            images: List of numpy arrays [(H, W, 3) BGR, ...]
            confidence_threshold: Minimum confidence for detections (0-1)
            padding: Pixels to add around detected region
            batch_size: Batch size for TensorRT inference (32, 64, or 128)

        Returns:
            List of detection lists [
                [{'bbox': (x1, y1, x2, y2), 'confidence': conf}, ...],  # Frame 0
                [{'bbox': (x1, y1, x2, y2), 'confidence': conf}, ...],  # Frame 1
                ...
            ]
        """
        if not self.use_yolo:
            return [[] for _ in images]

        if not images:
            return []

        # Determine device
        if self._require_tensorrt:
            if torch is None or not torch.cuda.is_available():
                raise RuntimeError(
                    "TensorRT mode requires CUDA GPU but torch.cuda.is_available() is False."
                )
            device_arg = 0  # GPU only
        else:
            device_arg = 0 if (torch is not None and torch.cuda.is_available()) else 'cpu'

        all_detections = []

        # 🔥 CRITICAL: Pad all images to 640x640 for TensorRT batch engine
        # TensorRT engine expects square 640x640 inputs
        padded_images = []
        pad_info = []  # Store (scale, pad_left, pad_top, orig_w, orig_h) for each image

        for img in images:
            h, w = img.shape[:2]

            # Calculate scale to fit in 640x640
            scale = min(640.0 / w, 640.0 / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            # Resize maintaining aspect ratio
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Calculate padding to make it 640x640
            pad_w = 640 - new_w
            pad_h = 640 - new_h
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left

            # Pad with gray (114, 114, 114) - YOLO standard
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                       cv2.BORDER_CONSTANT, value=(114, 114, 114))

            padded_images.append(padded)
            pad_info.append((scale, left, top, w, h))

        # Process in batches
        for i in range(0, len(padded_images), batch_size):
            batch = padded_images[i:i + batch_size]
            batch_info = pad_info[i:i + batch_size]
            orig_images = images[i:i + batch_size]

            # Run batch inference on 640x640 padded images (EXTREME SPEED!)
            results = self.model(batch, conf=confidence_threshold, device=device_arg,
                               verbose=False, imgsz=640)

            # Process each result in the batch
            for orig_img, result, (scale, pad_left, pad_top, orig_w, orig_h) in zip(orig_images, results, batch_info):
                bboxes = []

                boxes = result.boxes
                for box in boxes:
                    # Get bounding box coordinates (in padded 640x640 space)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                    # Remove padding offset
                    x1 = x1 - pad_left
                    y1 = y1 - pad_top
                    x2 = x2 - pad_left
                    y2 = y2 - pad_top

                    # Scale back to original image coordinates
                    x1 = x1 / scale
                    y1 = y1 / scale
                    x2 = x2 / scale
                    y2 = y2 / scale

                    # Add padding and clamp to image bounds
                    x1 = max(0, int(x1) - padding)
                    y1 = max(0, int(y1) - padding)
                    x2 = min(orig_w, int(x2) + padding)
                    y2 = min(orig_h, int(y2) + padding)

                    conf = float(box.conf[0])

                    bboxes.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': conf
                    })

                all_detections.append(bboxes)

        return all_detections

    def create_mask(
        self,
        image,
        detections,
        expand_ratio=0.0,
        expand_pixels=0,
        feather_pixels=21,
    ):
        """
        Create binary mask from detections

        Args:
            image: numpy array (H, W, 3)
            detections: List of detection dicts from detect()
            expand_ratio: fractional expansion applied to bbox (value added per side)
            expand_pixels: minimum pixel expansion applied per side
            feather_pixels: Gaussian blur kernel size (odd) for smooth edges

        Returns:
            Binary mask (H, W) uint8
        """
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        expand_ratio = float(max(expand_ratio, 0.0))
        expand_pixels = int(max(expand_pixels, 0))

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)

            pad_x = max(expand_pixels, int(width * expand_ratio * 0.5))
            pad_y = max(expand_pixels, int(height * expand_ratio * 0.5))

            x1_exp = max(0, x1 - pad_x)
            x2_exp = min(w, x2 + pad_x)
            y1_exp = max(0, y1 - pad_y)
            y2_exp = min(h, y2 + pad_y)

            mask[y1_exp:y2_exp, x1_exp:x2_exp] = 255

        if feather_pixels > 0 and mask.any():
            k = int(feather_pixels)
            if k % 2 == 0:
                k += 1  # kernel must be odd
            blurred = cv2.GaussianBlur(mask, (k, k), 0)
            _, mask = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

        return mask


# Test
if __name__ == "__main__":
    print("Testing YOLOv8 watermark detection...")

    image = cv2.imread('test_frame.jpg')
    if image is None:
        print("Need test_frame.jpg!")
        exit()

    detector = YOLOWatermarkDetector()

    print("\nDetecting watermarks...")
    detections = detector.detect(image, confidence_threshold=0.3)

    if not detections:
        print("No watermarks detected!")
    else:
        print(f"\nFound {len(detections)} watermark(s)")

        # Create mask
        mask = detector.create_mask(image, detections)
        cv2.imwrite('mask_yolo.png', mask)
        print("Saved mask_yolo.png")

        # Visualize detections
        vis = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(vis, f"{conf:.2%}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imwrite('detection_yolo.jpg', vis)
        print("Saved detection_yolo.jpg")
