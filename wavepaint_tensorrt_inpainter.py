import sys
import os
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
import torch

class WavePaintTensorRTInpainter:
    def __init__(self, engine_path='weights/wavepaint.engine'):
        """Initialize WavePaint TensorRT inpainter for fast GPU inference"""
        self.engine_path = engine_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.device == 'cpu':
            print("❌ CUDA not available! WavePaint TensorRT requires GPU")
            raise RuntimeError("CUDA required for TensorRT")

        if not os.path.exists(engine_path):
            print(f"❌ TensorRT engine not found: {engine_path}")
            print("Run EXPORT_WAVEPAINT_TENSORRT_SIMPLE.bat first")
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        try:
            print(f"Loading WavePaint TensorRT engine from {engine_path}...")
            import tensorrt as trt

            # Load TensorRT engine
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)

            with open(engine_path, 'rb') as f:
                engine_data = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_data)

            self.context = self.engine.create_execution_context()

            # Allocate GPU buffers using torch CUDA tensors (no pycuda needed!)
            self.d_input_img = torch.zeros((1, 3, 256, 256), dtype=torch.float32).cuda()
            self.d_input_mask = torch.zeros((1, 1, 256, 256), dtype=torch.float32).cuda()
            self.d_output = torch.zeros((1, 3, 256, 256), dtype=torch.float32).cuda()

            # Create bindings list for TensorRT
            self.bindings = [
                int(self.d_input_img.data_ptr()),
                int(self.d_input_mask.data_ptr()),
                int(self.d_output.data_ptr())
            ]

            print("✅ WavePaint TensorRT engine loaded successfully!")
            print("   Expected speed: 10-20x faster than PyTorch WavePaint")
            print()

        except ImportError:
            print("❌ TensorRT not installed!")
            raise
        except Exception as e:
            print(f"❌ Failed to load TensorRT engine: {e}")
            import traceback
            traceback.print_exc()
            raise

    def inpaint_region(self, image, mask, crop_size=512):
        """
        Remove watermark using WavePaint TensorRT with HIGH QUALITY crop-based processing!

        Args:
            image: numpy array (H, W, 3) BGR
            mask: numpy array (H, W) where 255 = area to inpaint
            crop_size: Size of crop region around watermark (default 512 for 2x downscale vs 8x full frame)

        Returns:
            numpy array (H, W, 3) BGR with watermark removed
        """
        h, w = image.shape[:2]

        # Find bounding box of mask
        mask_coords = np.where(mask > 127)
        if len(mask_coords[0]) == 0:
            # No mask, return original
            return image

        y_min, y_max = mask_coords[0].min(), mask_coords[0].max()
        x_min, x_max = mask_coords[1].min(), mask_coords[1].max()

        # Calculate center of watermark
        center_y = (y_min + y_max) // 2
        center_x = (x_min + x_max) // 2

        # Create square crop centered on watermark with extra context
        half_crop = crop_size // 2
        crop_y1 = max(0, center_y - half_crop)
        crop_y2 = min(h, center_y + half_crop)
        crop_x1 = max(0, center_x - half_crop)
        crop_x2 = min(w, center_x + half_crop)

        # Ensure crop is square (adjust if hit boundaries)
        crop_h = crop_y2 - crop_y1
        crop_w = crop_x2 - crop_x1
        if crop_h != crop_w:
            target_size = min(crop_h, crop_w)
            crop_y2 = crop_y1 + target_size
            crop_x2 = crop_x1 + target_size

        # Extract crop and crop mask
        img_crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        mask_crop = mask[crop_y1:crop_y2, crop_x1:crop_x2]

        # Resize crop to 256x256 (much better than 1920→256!)
        # Downscale ratio: 512→256 (2x) vs 1920→256 (8x) = 4x better quality!
        img_256 = cv2.resize(img_crop, (256, 256))
        mask_256 = cv2.resize(mask_crop, (256, 256))

        # Convert to float and normalize
        img_float = img_256.astype(np.float32) / 255.0
        mask_float = mask_256.astype(np.float32) / 255.0

        # CelebHQ model expects pre-masked image (black out watermark area)
        masked_img = img_float * (1 - mask_float[:, :, np.newaxis])

        # Convert to tensors (CHW format)
        img_input = masked_img.transpose(2, 0, 1)  # HWC -> CHW
        mask_input = mask_float[np.newaxis, :, :]  # Add channel dim

        # Copy to GPU tensors
        self.d_input_img.copy_(torch.from_numpy(img_input).unsqueeze(0))
        self.d_input_mask.copy_(torch.from_numpy(mask_input).unsqueeze(0))

        # Run TensorRT inference
        self.context.execute_v2(self.bindings)

        # Get result from GPU
        result_256 = self.d_output.cpu().numpy()[0]  # Remove batch dim
        result_256 = result_256.transpose(1, 2, 0)  # CHW -> HWC
        result_256 = (result_256 * 255).clip(0, 255).astype(np.uint8)

        # Resize back to crop size using CUBIC for best quality
        crop_h_actual = crop_y2 - crop_y1
        crop_w_actual = crop_x2 - crop_x1
        result_crop = cv2.resize(result_256, (crop_w_actual, crop_h_actual), interpolation=cv2.INTER_CUBIC)

        # Paste inpainted crop back into original frame
        result = image.copy()
        result[crop_y1:crop_y2, crop_x1:crop_x2] = result_crop

        return result
