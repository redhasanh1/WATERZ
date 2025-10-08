import sys
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
try:
    import torch
except Exception:
    torch = None

class YOLOWatermarkDetector:
    def __init__(self, model_path=None):
        """
        Initialize YOLOv8 watermark detector

        Args:
            model_path: Path to custom YOLOv8 model, or None to use trained Sora model
        """
        try:
            print("Loading YOLOv8 watermark detection model...")
            from ultralytics import YOLO
            import os

            if model_path:
                # Use custom model
                self.model = YOLO(model_path)
                print(f"✅ Loaded custom model: {model_path}")
            else:
                # Try to use TensorRT engine first (FAST!)
                tensorrt_paths = [
                    'runs/detect/new_sora_watermark/weights/best.engine',  # NEW trained model
                    '../runs/detect/new_sora_watermark/weights/best.engine',
                    'D:/github/RoomFinderAI/watermarkz/runs/detect/new_sora_watermark/weights/best.engine',
                    'runs/detect/sora_watermark/weights/best.engine',      # Old model
                    'yolov8n.engine',
                ]

                tensorrt_model = None
                for path in tensorrt_paths:
                    if os.path.exists(path):
                        tensorrt_model = path
                        break

                tensorrt_loaded = False
                if tensorrt_model:
                    if (torch is None) or (not torch.cuda.is_available()):
                        print("⚠️  CUDA torch not available in this process; skipping YOLO TensorRT engine")
                    else:
                        try:
                            print(f"🚀 Attempting to load TensorRT engine: {tensorrt_model}")
                            # Try to load TensorRT
                            self.model = YOLO(tensorrt_model, task='detect')
                            print(f"✅ TensorRT engine loaded! (20-35 fps on GTX 1660 Ti)")
                            tensorrt_loaded = True
                        except FileNotFoundError as e:
                            if 'nvinfer' in str(e):
                                print(f"⚠️  TensorRT DLLs not found (nvinfer_10.dll missing)")
                                print(f"   Falling back to PyTorch .pt model (slower but works)")
                            else:
                                print(f"⚠️  TensorRT load error: {e}")
                            tensorrt_loaded = False
                        except Exception as e:
                            print(f"⚠️  TensorRT failed: {e}")
                            print(f"   Falling back to PyTorch model...")
                            tensorrt_loaded = False

                if not tensorrt_loaded:
                    # Fallback to .pt model
                    print("⚠️  Using .pt model (10-15 fps, slower than TensorRT)")

                    # Try to use trained Sora watermark model first
                    # Check multiple possible paths (prioritize new_sora_watermark)
                    possible_paths = [
                        'runs/detect/new_sora_watermark/weights/best.pt',  # NEW trained model
                        '../runs/detect/new_sora_watermark/weights/best.pt',
                        'D:/github/RoomFinderAI/watermarkz/runs/detect/new_sora_watermark/weights/best.pt',
                        '/workspaces/RoomFinderAI/watermarkz/runs/detect/new_sora_watermark/weights/best.pt',
                        'runs/detect/sora_watermark/weights/best.pt',  # Old model fallback
                        '../runs/detect/sora_watermark/weights/best.pt',
                        'D:/github/RoomFinderAI/runs/detect/sora_watermark/weights/best.pt',
                        '/workspaces/RoomFinderAI/runs/detect/sora_watermark/weights/best.pt',
                    ]

                    sora_model = None
                    for path in possible_paths:
                        if os.path.exists(path):
                            sora_model = path
                            break

                    if sora_model:
                        print("Loading trained Sora watermark model...")
                        self.model = YOLO(sora_model)
                        print(f"✅ Loaded trained Sora model: {sora_model}")
                    else:
                        print("⚠️  Trained Sora model not found!")
                        print("Checked paths:")
                        for p in possible_paths:
                            print(f"  - {p}")
                        print("\nRun TRAIN_YOLO.bat to train the model first")
                        print("\nFalling back to generic watermark detector...")

                        # Try to download from Hugging Face
                        try:
                            from huggingface_hub import hf_hub_download

                            model_file = hf_hub_download(
                                repo_id="qfisch/yolov8n-watermark-detection",
                                filename="best.pt"
                            )
                            self.model = YOLO(model_file)
                            print("✅ Loaded qfisch/yolov8n-watermark-detection model")

                        except Exception as e:
                            print(f"Could not download from HuggingFace: {e}")
                            print("Falling back to YOLOv8n base model")
                            self.model = YOLO('yolov8n.pt')

            self.use_yolo = True
            print("✅ YOLOv8 ready for watermark detection!")

        except Exception as e:
            print(f"❌ Could not load YOLOv8: {e}")
            print("Will use fallback detection")
            self.use_yolo = False

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
