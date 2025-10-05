"""
Optimized YOLO Detector for Production
- GPU acceleration with CUDA
- TensorRT support for 3-5x speedup
- FP16 precision for 2x speedup
- Batch processing support
"""

import torch
from ultralytics import YOLO
import cv2
import numpy as np
import time


class OptimizedYOLODetector:
    def __init__(self, model_path='yolov8n.pt', use_tensorrt=False):
        """
        Optimized YOLO detector for production watermark detection

        Args:
            model_path: Path to YOLO model (.pt file)
            use_tensorrt: Use TensorRT for 3-5x speedup (requires export first)

        Performance:
            CPU: 1-2 images/sec
            GPU (FP32): 15-30 images/sec
            GPU (FP16): 30-60 images/sec
            GPU (TensorRT): 50-100 images/sec
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_half = self.device == 'cuda'

        print("=" * 60)
        print("Initializing Optimized YOLO Detector")
        print("=" * 60)

        if use_tensorrt and self.device == 'cuda':
            # Use TensorRT optimized model (fastest - 3-5x speedup)
            engine_path = model_path.replace('.pt', '.engine')
            try:
                self.model = YOLO(engine_path)
                print(f"✅ Loaded TensorRT model: {engine_path}")
                print(f"   Device: {self.device}")
                print(f"   Expected speed: 50-100 images/sec")
            except Exception as e:
                print(f"⚠️  TensorRT model not found: {e}")
                print(f"   To create: yolo export model={model_path} format=engine device=0 half=True")
                print(f"   Falling back to standard model...")
                use_tensorrt = False

        if not use_tensorrt:
            # Use standard PyTorch model
            self.model = YOLO(model_path)
            self.model.to(self.device)

            # Enable half precision (FP16) for 2x speedup on GPU
            if self.use_half:
                try:
                    self.model.half()
                    print(f"✅ Loaded FP16 model: {model_path}")
                    print(f"   Device: {self.device}")
                    print(f"   Expected speed: 30-60 images/sec")
                except:
                    self.use_half = False
                    print(f"⚠️  FP16 not supported, using FP32")
            else:
                print(f"⚠️  CPU mode (SLOW): {model_path}")
                print(f"   Expected speed: 1-2 images/sec")
                print(f"   Install CUDA for 10-50x speedup!")

        # GPU info
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"   GPU: {gpu_name}")
            print(f"   VRAM: {gpu_memory:.1f} GB")

            # Limit GPU memory usage to keep PC usable
            # GTX 1660 Ti (6GB): Use 70% = 4.2GB (leaves 1.8GB for games/apps)
            # RTX 4090 (24GB): Use 80% = 19.2GB (leaves 4.8GB free)
            memory_fraction = 0.7 if gpu_memory <= 8 else 0.8
            torch.cuda.set_per_process_memory_fraction(memory_fraction, device=0)
            print(f"   GPU Memory Limit: {int(memory_fraction*100)}% ({gpu_memory * memory_fraction:.1f} GB)")

        # Warmup GPU (first inference is always slow)
        print("   Warming up model...")
        start = time.time()
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy_img, verbose=False)
        warmup_time = (time.time() - start) * 1000
        print(f"   Warmup complete: {warmup_time:.1f}ms")

        print("=" * 60)

    def detect(self, image, confidence_threshold=0.3, padding=30):
        """
        Detect watermarks with GPU acceleration

        Args:
            image: Input image (numpy array)
            confidence_threshold: Minimum confidence for detection
            padding: Pixels to add around detected watermark

        Returns:
            List of detections with bbox and confidence
        """
        # Run inference with GPU acceleration
        results = self.model.predict(
            image,
            conf=confidence_threshold,
            device=self.device,
            half=self.use_half,
            verbose=False,
            imgsz=640  # Resize to 640x640 for speed
        )

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()

                # Add padding around watermark
                x1 = max(0, int(x1) - padding)
                y1 = max(0, int(y1) - padding)
                x2 = min(image.shape[1], int(x2) + padding)
                y2 = min(image.shape[0], int(y2) + padding)

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf
                })

        return detections

    def detect_batch(self, images, confidence_threshold=0.3, padding=30):
        """
        Batch detection for 2-3x throughput

        Args:
            images: List of images
            confidence_threshold: Minimum confidence
            padding: Padding around watermarks

        Returns:
            List of detection lists (one per image)
        """
        # Batch inference
        results = self.model.predict(
            images,
            conf=confidence_threshold,
            device=self.device,
            half=self.use_half,
            verbose=False,
            imgsz=640
        )

        all_detections = []
        for result, image in zip(results, images):
            detections = []
            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().item()

                # Add padding
                x1 = max(0, int(x1) - padding)
                y1 = max(0, int(y1) - padding)
                x2 = min(image.shape[1], int(x2) + padding)
                y2 = min(image.shape[0], int(y2) + padding)

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf
                })

            all_detections.append(detections)

        return all_detections

    def create_mask(self, image, detections):
        """
        Create binary mask for inpainting

        Args:
            image: Input image
            detections: List of detections from detect()

        Returns:
            Binary mask (255 where watermark detected, 0 elsewhere)
        """
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            mask[y1:y2, x1:x2] = 255

        return mask

    def benchmark(self, image=None, num_runs=100):
        """
        Benchmark inference speed

        Args:
            image: Test image (uses dummy if None)
            num_runs: Number of inferences to average

        Returns:
            Dict with performance metrics
        """
        if image is None:
            image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        print(f"Benchmarking {num_runs} inferences...")

        # Warmup
        for _ in range(10):
            self.detect(image)

        # Benchmark
        start = time.time()
        for _ in range(num_runs):
            self.detect(image)
        total_time = time.time() - start

        avg_time_ms = (total_time / num_runs) * 1000
        throughput = num_runs / total_time

        results = {
            'avg_time_ms': avg_time_ms,
            'throughput': throughput,
            'device': self.device,
            'half_precision': self.use_half,
            'total_time_sec': total_time
        }

        print(f"\n{'='*60}")
        print(f"Benchmark Results ({num_runs} runs)")
        print(f"{'='*60}")
        print(f"Average time: {avg_time_ms:.2f}ms per image")
        print(f"Throughput: {throughput:.1f} images/sec")
        print(f"Device: {self.device}")
        print(f"Half precision: {self.use_half}")
        print(f"{'='*60}\n")

        return results


# Test/Demo
if __name__ == "__main__":
    print("\n🚀 Testing Optimized YOLO Detector\n")

    # Initialize detector
    detector = OptimizedYOLODetector(
        model_path='yolov8n.pt',
        use_tensorrt=False  # Set to True if you've exported to TensorRT
    )

    # Test image
    test_image_path = 'test_frame.jpg'
    if not os.path.exists(test_image_path):
        print(f"Creating dummy test image...")
        test_image = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    else:
        test_image = cv2.imread(test_image_path)
        print(f"Loaded test image: {test_image_path}")

    # Single detection
    print("\n🔍 Testing single image detection...")
    start = time.time()
    detections = detector.detect(test_image)
    elapsed = (time.time() - start) * 1000
    print(f"   Detections: {len(detections)}")
    print(f"   Time: {elapsed:.1f}ms")

    # Batch detection
    print("\n📦 Testing batch detection (10 images)...")
    images = [test_image] * 10
    start = time.time()
    batch_detections = detector.detect_batch(images)
    elapsed = (time.time() - start) * 1000
    print(f"   Total time: {elapsed:.1f}ms")
    print(f"   Time per image: {elapsed/10:.1f}ms")
    print(f"   Throughput: {10000/elapsed:.1f} images/sec")

    # Benchmark
    print("\n⚡ Running full benchmark...")
    detector.benchmark(test_image, num_runs=100)

    print("\n✅ All tests complete!\n")
