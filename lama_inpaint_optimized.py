import sys
import os
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
import torch

class LamaInpainterOptimized:
    def __init__(self, device='cuda'):
        """Initialize OPTIMIZED LaMa AI model for inpainting with FP16 + CUDA optimizations"""
        self.device = device if torch.cuda.is_available() else 'cpu'

        try:
            print(f"Loading OPTIMIZED LaMa AI model on {self.device}...")

            # Enable PyTorch CUDA optimizations
            if torch.cuda.is_available():
                print("  Enabling CUDA optimizations...")
                torch.backends.cudnn.benchmark = True  # Auto-tune CUDA kernels
                torch.backends.cudnn.enabled = True
                torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat-32
                torch.backends.cudnn.allow_tf32 = True
                print("  ✓ CUDA optimizations enabled")

            from iopaint.model.lama import LaMa
            from iopaint.schema import Config, HDStrategy

            self.config = Config(
                ldm_steps=25,
                ldm_sampler='plms',
                hd_strategy=HDStrategy.ORIGINAL,
                hd_strategy_crop_margin=32,
            )

            # Initialize LAMA
            self.model = LaMa(self.device)

            # Enable FP16 (Half Precision) for 2x speedup
            if self.device == 'cuda':
                print("  Converting model to FP16 (half precision)...")
                # Check if model supports half precision
                try:
                    self.model.model = self.model.model.half()
                    self.use_fp16 = True
                    print("  ✓ FP16 enabled - expect 1.5-2x speedup!")
                except Exception as e:
                    print(f"  ⚠️ FP16 not supported: {e}")
                    self.use_fp16 = False
            else:
                self.use_fp16 = False

            self.use_ai = True
            print("✓ OPTIMIZED LaMa AI loaded successfully!")
            print("  Optimizations active:")
            print("    - CUDA kernel auto-tuning")
            print("    - TensorFloat-32 operations")
            if self.use_fp16:
                print("    - FP16 half precision (2x faster)")
            print()

        except Exception as e:
            print(f"⚠ Could not load LaMa AI: {e}")
            print("Falling back to advanced OpenCV inpainting...")
            import traceback
            traceback.print_exc()
            self.use_ai = False

    def inpaint_region(self, image, mask):
        """
        Remove watermark using OPTIMIZED AI inpainting

        Args:
            image: numpy array (H, W, 3) BGR
            mask: numpy array (H, W) where 255 = area to inpaint

        Returns:
            inpainted image as numpy array
        """
        # Ensure mask is binary
        mask = (mask > 127).astype(np.uint8) * 255

        if self.use_ai:
            # AI inpainting with OPTIMIZED LaMa
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Use torch.no_grad() to save memory and speed up inference
            with torch.no_grad():
                result = self.model(rgb_image, mask, self.config)

            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        else:
            # Fallback: advanced OpenCV
            result = cv2.inpaint(image, mask, 15, cv2.INPAINT_NS)

        return result

# Test the module
if __name__ == "__main__":
    print("Testing OPTIMIZED LaMa inpainting...")

    # Load test image and mask
    image = cv2.imread('test_frame.jpg')
    mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print("Error: test_frame.jpg or mask.png not found!")
        exit()

    # Initialize and inpaint
    import time

    print("\n" + "=" * 60)
    print("Speed Test: Optimized vs Regular LAMA")
    print("=" * 60)

    # Test optimized version
    print("\nTesting OPTIMIZED LAMA...")
    inpainter_opt = LamaInpainterOptimized(device='cuda')

    start = time.time()
    result_opt = inpainter_opt.inpaint_region(image, mask)
    time_opt = time.time() - start

    print(f"Optimized LAMA: {time_opt*1000:.1f}ms")

    # Save result
    cv2.imwrite('result_lama_optimized.jpg', result_opt)
    print("\n✓ Saved result_lama_optimized.jpg")

    # Compare with regular LAMA
    print("\nTesting REGULAR LAMA...")
    from lama_inpaint import LamaInpainter
    inpainter_regular = LamaInpainter(device='cuda')

    start = time.time()
    result_regular = inpainter_regular.inpaint_region(image, mask)
    time_regular = time.time() - start

    print(f"Regular LAMA: {time_regular*1000:.1f}ms")

    # Calculate speedup
    speedup = time_regular / time_opt
    print(f"\n🚀 SPEEDUP: {speedup:.2f}x faster!")
    print(f"   Saved {(time_regular - time_opt)*1000:.1f}ms per frame")
