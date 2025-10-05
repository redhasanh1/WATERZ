import sys
import os
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
import torch

class LamaInpainter:
    def __init__(self, device='cuda'):
        """Initialize LaMa AI model for inpainting"""
        self.device = device if torch.cuda.is_available() else 'cpu'

        try:
            print(f"Loading LaMa AI model on {self.device}...")
            from iopaint.model.lama import LaMa
            from iopaint.schema import Config, HDStrategy

            self.config = Config(
                ldm_steps=25,
                ldm_sampler='plms',
                hd_strategy=HDStrategy.ORIGINAL,
                hd_strategy_crop_margin=32,
            )

            self.model = LaMa(self.device)
            self.use_ai = True
            print("✓ LaMa AI loaded successfully!")

        except Exception as e:
            print(f"⚠ Could not load LaMa AI: {e}")
            print("Falling back to advanced OpenCV inpainting...")
            self.use_ai = False

    def inpaint_region(self, image, mask):
        """
        Remove watermark using AI or advanced inpainting

        Args:
            image: numpy array (H, W, 3) BGR
            mask: numpy array (H, W) where 255 = area to inpaint

        Returns:
            inpainted image as numpy array
        """
        # Ensure mask is binary
        mask = (mask > 127).astype(np.uint8) * 255

        if self.use_ai:
            # AI inpainting with LaMa
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = self.model(rgb_image, mask, self.config)
            result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        else:
            # Fallback: advanced OpenCV
            result = cv2.inpaint(image, mask, 15, cv2.INPAINT_NS)

        return result

# Test the module
if __name__ == "__main__":
    print("Testing LaMa inpainting...")

    # Load test image and mask
    image = cv2.imread('test_frame.jpg')
    mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print("Error: test_frame.jpg or mask.png not found!")
        exit()

    # Initialize and inpaint
    inpainter = LamaInpainter(device='cuda')
    result = inpainter.inpaint_region(image, mask)

    # Save result
    cv2.imwrite('result_lama.jpg', result)
    print("Saved result_lama.jpg - AI inpainted result!")
