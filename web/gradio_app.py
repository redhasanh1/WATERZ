#!/usr/bin/env python3
"""
Gradio Web Interface for AI Watermark Remover
"""

import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'python_packages'))

import gradio as gr
import cv2
import numpy as np
from yolo_detector import YOLOWatermarkDetector
from lama_inpaint_local import LamaInpainter
import tempfile

print("=" * 60)
print("AI Watermark Remover - Gradio Web Interface")
print("=" * 60)

# Initialize detector and inpainter
print("Initializing YOLO detector...")
detector = YOLOWatermarkDetector()

print("Initializing LaMa inpainter...")
inpainter = LamaInpainter()

print("Models loaded successfully!")
print("=" * 60)


def process_image(image):
    """Process a single image to remove watermark"""
    if image is None:
        return None, "No image uploaded"

    try:
        # Convert from RGB to BGR for OpenCV
        frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Detect watermark
        detections = detector.detect(frame, confidence_threshold=0.3, padding=30)

        if not detections:
            return image, "⚠️ No watermark detected. Returning original image."

        # Create mask
        mask = detector.create_mask(frame, detections)

        # Remove watermark
        result = inpainter.inpaint_region(frame, mask)

        # Convert back to RGB for display
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        return result_rgb, f"✅ Watermark removed! Detected {len(detections)} watermark(s)."

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def process_video(video_path):
    """Process video to remove watermark from all frames"""
    if video_path is None:
        return None, "No video uploaded"

    try:
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, "❌ Failed to open video"

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create temporary output file
        output_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            return None, "❌ Failed to create video writer"

        # Load template for fallback
        template_path = os.path.join(parent_dir, 'watermark_template.png')
        template = None
        if os.path.exists(template_path):
            template = cv2.imread(template_path)

        last_valid_bbox = None
        frames_processed = 0
        frames_with_watermark = 0

        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect watermark
            detections = detector.detect(frame, confidence_threshold=0.3, padding=30)

            # Fallback: template matching if YOLO missed
            if not detections and template is not None and last_valid_bbox:
                th, tw = template.shape[:2]
                result_match = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result_match)

                if max_val > 0.6:
                    x1, y1 = max_loc
                    x2, y2 = x1 + tw, y1 + th
                    x1 = max(0, x1 - 30)
                    y1 = max(0, y1 - 30)
                    x2 = min(width, x2 + 30)
                    y2 = min(height, y2 + 30)
                    detections = [{'bbox': (x1, y1, x2, y2), 'confidence': max_val}]

            # Use last known position as fallback
            if not detections and last_valid_bbox:
                detections = [{'bbox': last_valid_bbox, 'confidence': 0.0}]

            if detections:
                frames_with_watermark += 1

                # Update last known position
                if detections[0]['confidence'] > 0.3:
                    last_valid_bbox = detections[0]['bbox']

                # Create mask and remove watermark
                mask = detector.create_mask(frame, detections)
                try:
                    processed_frame = inpainter.inpaint_region(frame, mask)
                    out.write(processed_frame)
                except Exception as e:
                    out.write(frame)
            else:
                out.write(frame)

            frames_processed += 1

        # Cleanup
        cap.release()
        out.release()

        status = f"✅ Video processing complete!\n"
        status += f"Frames processed: {frames_processed}\n"
        status += f"Watermarks detected: {frames_with_watermark}"

        return output_path, status

    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# Custom CSS for dark theme with purple gradients
custom_css = """
.gradio-container {
    background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%) !important;
}

.gr-button-primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
}

.gr-button-primary:hover {
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4) !important;
}

h1, h2, h3, .gr-label {
    color: #ffffff !important;
}

.gr-form {
    background: rgba(26, 26, 46, 0.8) !important;
    border: 1px solid rgba(102, 126, 234, 0.2) !important;
    border-radius: 16px !important;
}
"""

# Create Gradio interface with tabs
with gr.Blocks(css=custom_css, title="AI Watermark Remover", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎨 AI Watermark Remover
        ### Remove watermarks from images and videos using YOLO + LaMa AI

        Upload your file below and click **Process** to remove watermarks automatically.
        """
    )

    with gr.Tabs():
        # Image Processing Tab
        with gr.TabItem("📸 Image"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        label="Upload Image",
                        type="numpy",
                        height=400
                    )
                    image_btn = gr.Button("🚀 Remove Watermark", variant="primary", size="lg")

                with gr.Column():
                    image_output = gr.Image(
                        label="Result",
                        type="numpy",
                        height=400
                    )
                    image_status = gr.Textbox(label="Status", lines=2)

            gr.Examples(
                examples=[
                    os.path.join(parent_dir, "test_frame.jpg") if os.path.exists(os.path.join(parent_dir, "test_frame.jpg")) else None
                ],
                inputs=image_input,
                label="Example Images"
            )

            image_btn.click(
                fn=process_image,
                inputs=image_input,
                outputs=[image_output, image_status]
            )

        # Video Processing Tab
        with gr.TabItem("🎬 Video"):
            with gr.Row():
                with gr.Column():
                    video_input = gr.Video(
                        label="Upload Video",
                        height=400
                    )
                    video_btn = gr.Button("🚀 Remove Watermark", variant="primary", size="lg")

                with gr.Column():
                    video_output = gr.Video(
                        label="Result",
                        height=400
                    )
                    video_status = gr.Textbox(label="Status", lines=3)

            gr.Markdown("⚠️ **Note:** Video processing may take several minutes depending on length.")

            video_btn.click(
                fn=process_video,
                inputs=video_input,
                outputs=[video_output, video_status]
            )

    gr.Markdown(
        """
        ---
        ### How it works:
        1. **Upload** your image or video with watermarks
        2. **Click** the process button
        3. **Download** your watermark-free result!

        **Powered by:** YOLOv8 Detection + LaMa Inpainting
        """
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
