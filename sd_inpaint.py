import sys
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
import torch

class SDInpainter:
    def __init__(self, device='cuda'):
        """Initialize Stable Diffusion inpainting - HitPaw quality"""
        self.device = device if torch.cuda.is_available() else 'cpu'

        try:
            print(f"Loading Stable Diffusion inpainting model on {self.device}...")
            print("This may take a minute to download the model...")

            from diffusers import StableDiffusionInpaintPipeline

            # Use SD 1.5 inpainting model
            model_id = "runwayml/stable-diffusion-inpainting"

            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            )
            self.pipe = self.pipe.to(self.device)

            # Enable memory optimizations
            if self.device == 'cuda':
                self.pipe.enable_attention_slicing()
                self.pipe.enable_vae_slicing()

            self.use_sd = True
            print("✅ Stable Diffusion loaded! HitPaw-quality inpainting ready!")

        except Exception as e:
            print(f"❌ Could not load SD: {e}")
            print("Falling back to advanced OpenCV...")
            self.use_sd = False

    def inpaint_region(self, image, mask):
        """
        Remove watermark using Stable Diffusion AI

        Args:
            image: numpy array (H, W, 3) BGR
            mask: numpy array (H, W) where 255 = area to inpaint

        Returns:
            inpainted image
        """
        from PIL import Image

        # Ensure mask is binary
        mask = (mask > 127).astype(np.uint8) * 255

        if self.use_sd:
            # Convert to PIL for SD
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            pil_mask = Image.fromarray(mask)

            # SD inpainting with prompt
            prompt = "clean background, no text, no watermark, high quality"
            negative_prompt = "watermark, text, logo, blur, artifacts"

            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pil_image,
                mask_image=pil_mask,
                num_inference_steps=30,
                guidance_scale=7.5,
            ).images[0]

            # Convert back to BGR numpy
            result_np = np.array(result)
            result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)

            return result_bgr

        else:
            # Fallback
            return cv2.inpaint(image, mask, 15, cv2.INPAINT_NS)


# Test
if __name__ == "__main__":
    print("Testing Stable Diffusion inpainting...")

    image = cv2.imread('test_frame.jpg')
    mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print("Need test_frame.jpg and mask.png!")
        exit()

    inpainter = SDInpainter(device='cuda')
    result = inpainter.inpaint_region(image, mask)

    cv2.imwrite('result_sd.jpg', result)
    print("\n✅ Saved result_sd.jpg - Check for HitPaw-quality removal!")

    # Comparison
    comparison = np.hstack([image, result])
    cv2.imwrite('comparison_sd.jpg', comparison)
    print("Saved comparison_sd.jpg (before | after)")
