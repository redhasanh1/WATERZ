import sys
import os

# Add python_packages to path
sys.path.insert(0, 'python_packages')

import cv2
import numpy as np
from tqdm import tqdm
import time

# Import detection and inpainting
from yolo_detector import YOLOWatermarkDetector
from lama_inpaint_local import LamaInpainter

def process_video(video_path):
    """Test video processing with YOLO + LaMa"""

    print("=" * 60)
    print("Loading AI Models (YOLO + LaMa)")
    print("=" * 60)

    start_time = time.time()

    # Initialize models
    detector = YOLOWatermarkDetector()
    inpainter = LamaInpainter()

    load_time = time.time() - start_time
    print(f"✅ Models loaded in {load_time:.2f} seconds\n")

    # Open video
    print("=" * 60)
    print("Opening Video")
    print("=" * 60)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Failed to open video!")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {total_frames/fps:.2f} seconds\n")

    # Setup output
    output_path = os.path.join('results', 'test_output.avi')
    os.makedirs('results', exist_ok=True)

    # Use XVID codec (built-in, works on Windows)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("❌ Failed to create video writer!")
        return

    print("=" * 60)
    print("Processing Video")
    print("=" * 60)
    print()

    frames_processed = 0
    frames_with_watermark = 0
    last_valid_bbox = None
    processing_times = []

    start_processing = time.time()

    # Process frames
    for frame_num in tqdm(range(total_frames), desc="Processing frames"):
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # Detect watermark
        detections = detector.detect(frame, confidence_threshold=0.3, padding=30)

        # Use temporal consistency (last known position)
        if not detections and last_valid_bbox:
            detections = [{'bbox': last_valid_bbox, 'confidence': 0.0}]

        if detections:
            frames_with_watermark += 1

            # Update last known position
            if detections[0]['confidence'] > 0.3:
                last_valid_bbox = detections[0]['bbox']

            # Remove watermark
            try:
                mask = detector.create_mask(frame, detections)
                processed_frame = inpainter.inpaint_region(frame, mask)
                out.write(processed_frame)
            except Exception as e:
                print(f"⚠️  Frame {frame_num} inpainting failed: {e}")
                out.write(frame)
        else:
            out.write(frame)

        frames_processed += 1

        # Track processing time
        frame_time = time.time() - frame_start
        processing_times.append(frame_time)

        # Print stats every 50 frames
        if frame_num > 0 and frame_num % 50 == 0:
            avg_time = sum(processing_times[-50:]) / len(processing_times[-50:])
            fps_current = 1.0 / avg_time if avg_time > 0 else 0
            print(f"\nFrame {frame_num}/{total_frames} | Speed: {fps_current:.2f} fps | Avg time: {avg_time*1000:.0f}ms/frame")

    # Cleanup
    cap.release()
    out.release()

    total_time = time.time() - start_processing
    avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
    avg_fps = 1.0 / avg_time if avg_time > 0 else 0

    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Frames processed: {frames_processed}")
    print(f"Watermarks detected: {frames_with_watermark}")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average speed: {avg_fps:.2f} frames/second")
    print(f"Average time per frame: {avg_time*1000:.0f}ms")
    print(f"\n📁 Output saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_video_processing.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    process_video(video_path)
