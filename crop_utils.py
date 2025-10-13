"""
Cropping utilities for watermark removal optimization
Based on faster-propainter's pre_post_process.py

Crops video to watermark region for 10-100x speedup on small watermarks
"""

import cv2
import numpy as np
import os
from typing import Tuple, List


def find_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Find the bounding box of non-zero pixels in a binary mask.

    Args:
        mask: Binary mask (H, W) with 0/255 values

    Returns:
        (min_x, max_x, min_y, max_y) coordinates of bounding box
    """
    binary_mask = np.where(mask > 0, 1, 0)
    nonzero_indices = np.nonzero(binary_mask)

    if len(nonzero_indices[0]) == 0 or len(nonzero_indices[1]) == 0:
        # No pixels, return zero box
        return (0, 0, 0, 0)

    y_coords = nonzero_indices[0]
    x_coords = nonzero_indices[1]

    min_x = np.min(x_coords)
    max_x = np.max(x_coords)
    min_y = np.min(y_coords)
    max_y = np.max(y_coords)

    return (min_x, max_x, min_y, max_y)


def calculate_crop_region(
    bbox: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    padding_ratio: float = 0.2,
    min_size: int = 128
) -> Tuple[int, int, int, int]:
    """
    Calculate optimal crop region for a watermark bounding box.

    Args:
        bbox: (x1, y1, x2, y2) watermark bounding box
        frame_width: Full frame width
        frame_height: Full frame height
        padding_ratio: Extra padding around bbox (0.2 = 20% on each side)
        min_size: Minimum crop dimension (for very small watermarks)

    Returns:
        (x, y, w, h) crop region coordinates
    """
    x1, y1, x2, y2 = bbox

    # Calculate bbox dimensions
    bbox_width = max(1, x2 - x1)
    bbox_height = max(1, y2 - y1)

    # Ensure minimum size
    target_width = max(min_size, bbox_width)
    target_height = max(min_size, bbox_height)

    # Add padding
    pad_x = int(target_width * padding_ratio)
    pad_y = int(target_height * padding_ratio)

    # Calculate crop region
    crop_x = max(0, x1 - pad_x)
    crop_y = max(0, y1 - pad_y)
    crop_w = min(target_width + 2 * pad_x, frame_width - crop_x)
    crop_h = min(target_height + 2 * pad_y, frame_height - crop_y)

    # Ensure dimensions are divisible by 16 for ProPainter
    crop_w = (crop_w // 16) * 16
    crop_h = (crop_h // 16) * 16

    # Adjust if we went out of bounds after alignment
    if crop_x + crop_w > frame_width:
        crop_x = frame_width - crop_w
    if crop_y + crop_h > frame_height:
        crop_y = frame_height - crop_h

    crop_x = max(0, crop_x)
    crop_y = max(0, crop_y)

    return (crop_x, crop_y, crop_w, crop_h)


def crop_frames_to_region(
    frames_dir: str,
    output_dir: str,
    crop_region: Tuple[int, int, int, int],
    frame_indices: List[int] = None
) -> int:
    """
    Crop frames from frames_dir to crop_region and save to output_dir.

    Args:
        frames_dir: Directory with frames (0000.png, 0001.png, ...)
        output_dir: Output directory for cropped frames
        crop_region: (x, y, w, h) region to crop
        frame_indices: Optional list of frame indices to process (None = all)

    Returns:
        Number of frames cropped
    """
    os.makedirs(output_dir, exist_ok=True)
    crop_x, crop_y, crop_w, crop_h = crop_region

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])

    if frame_indices is not None:
        frame_files = [f for i, f in enumerate(frame_files) if i in frame_indices]

    cropped_count = 0
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # Crop to region
        cropped = frame[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]

        # Save with same filename
        output_path = os.path.join(output_dir, frame_file)
        cv2.imwrite(output_path, cropped)
        cropped_count += 1

    return cropped_count


def crop_mask_to_region(
    mask: np.ndarray,
    crop_region: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    Crop a mask to the specified region.

    Args:
        mask: Full-size binary mask (H, W)
        crop_region: (x, y, w, h) region to crop

    Returns:
        Cropped mask (h, w)
    """
    crop_x, crop_y, crop_w, crop_h = crop_region
    return mask[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]


def merge_cleaned_region_back(
    original_frames_dir: str,
    cleaned_cropped_dir: str,
    output_dir: str,
    crop_region: Tuple[int, int, int, int],
    blend_border: int = 8
) -> int:
    """
    Merge cleaned cropped frames back into original full frames.

    Args:
        original_frames_dir: Directory with original full frames
        cleaned_cropped_dir: Directory with cleaned cropped frames
        output_dir: Output directory for merged frames
        crop_region: (x, y, w, h) region where cropped frames came from
        blend_border: Pixels to blend at crop borders for seamless merge

    Returns:
        Number of frames merged
    """
    os.makedirs(output_dir, exist_ok=True)
    crop_x, crop_y, crop_w, crop_h = crop_region

    original_files = sorted([f for f in os.listdir(original_frames_dir) if f.endswith('.png')])
    cleaned_files = sorted([f for f in os.listdir(cleaned_cropped_dir) if f.endswith('.png')])

    # Match frames by filename
    merged_count = 0
    for orig_file in original_files:
        if orig_file not in cleaned_files:
            # No cleaned version, copy original
            src = os.path.join(original_frames_dir, orig_file)
            dst = os.path.join(output_dir, orig_file)
            import shutil
            shutil.copy2(src, dst)
            merged_count += 1
            continue

        # Load frames
        original = cv2.imread(os.path.join(original_frames_dir, orig_file))
        cleaned_crop = cv2.imread(os.path.join(cleaned_cropped_dir, orig_file))

        if original is None or cleaned_crop is None:
            continue

        # Create output frame (copy of original)
        result = original.copy()

        # Simple paste (no blending for now - can add alpha blending later if needed)
        result[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w] = cleaned_crop

        # Save merged frame
        output_path = os.path.join(output_dir, orig_file)
        cv2.imwrite(output_path, result)
        merged_count += 1

    return merged_count


def estimate_speedup(
    original_resolution: Tuple[int, int],
    crop_resolution: Tuple[int, int]
) -> float:
    """
    Estimate processing speedup from cropping.

    Args:
        original_resolution: (width, height) of full frame
        crop_resolution: (width, height) of cropped region

    Returns:
        Estimated speedup factor (e.g., 50.0 = 50x faster)
    """
    orig_pixels = original_resolution[0] * original_resolution[1]
    crop_pixels = crop_resolution[0] * crop_resolution[1]

    if crop_pixels == 0:
        return 1.0

    # Speedup is roughly proportional to pixel count reduction
    # Add overhead factor (not perfectly linear)
    raw_speedup = orig_pixels / crop_pixels
    effective_speedup = raw_speedup * 0.7  # Account for overhead

    return effective_speedup


# Test/example
if __name__ == "__main__":
    # Example: Calculate crop region for a small watermark
    bbox = (100, 50, 200, 100)  # Small 100x50 watermark
    frame_size = (1920, 1080)

    crop_region = calculate_crop_region(bbox, frame_size[0], frame_size[1])
    print(f"Watermark bbox: {bbox}")
    print(f"Crop region: {crop_region}")
    print(f"Crop size: {crop_region[2]}x{crop_region[3]}")

    speedup = estimate_speedup(frame_size, (crop_region[2], crop_region[3]))
    print(f"Estimated speedup: {speedup:.1f}x")
