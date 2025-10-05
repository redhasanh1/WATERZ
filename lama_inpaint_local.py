import sys
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np

class LamaInpainter:
    def __init__(self):
        """Initialize LaMa inpainting - best for watermark removal"""
        try:
            print("Loading LaMa inpainting model...")
            from simple_lama_inpainting import SimpleLama
            from PIL import Image

            self.lama = SimpleLama()
            self.use_lama = True
            print("✅ LaMa loaded! Professional watermark removal ready!")

        except Exception as e:
            print(f"❌ Could not load LaMa: {e}")
            print("Falling back to OpenCV...")
            self.use_lama = False

    def inpaint_region(self, image, mask):
        """
        Remove watermark using LaMa AI

        Args:
            image: numpy array (H, W, 3) BGR
            mask: numpy array (H, W) where 255 = area to inpaint

        Returns:
            inpainted image
        """
        # Ensure mask is binary
        mask = (mask > 127).astype(np.uint8) * 255

        if self.use_lama:
            from PIL import Image

            # Convert to PIL RGB for LaMa
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            pil_mask = Image.fromarray(mask)

            # LaMa inpainting
            result = self.lama(pil_image, pil_mask)

            # Convert back to BGR numpy
            result_np = np.array(result)
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

            return result_bgr

        else:
            # Fallback
            return cv2.inpaint(image, mask, 15, cv2.INPAINT_NS)


# Test
if __name__ == "__main__":
    print("Testing LaMa inpainting...")

    image = cv2.imread('test_frame.jpg')
    mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print("Need test_frame.jpg and mask.png!")
        exit()

    inpainter = LamaInpainter()
    result = inpainter.inpaint_region(image, mask)

    cv2.imwrite('result_lama.jpg', result)
    print("\n✅ Saved result_lama.jpg - Check for professional removal!")

    # Comparison
    comparison = np.hstack([image, result])
    cv2.imwrite('comparison_lama.jpg', comparison)
    print("Saved comparison_lama.jpg (before | after)")
