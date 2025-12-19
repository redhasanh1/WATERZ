"""
GPU-Accelerated Static Mask Generator
Uses PyTorch/CUDA for fast mask generation from shape parameters

Usage:
    python static_mask_generator.py --shape '{"type":"rectangle","x":100,"y":100,"width":200,"height":100}' --width 1920 --height 1080 --output mask.png
"""
import torch
import numpy as np
import cv2
import os
import json
import argparse
import base64
from typing import Dict, List, Optional, Union


def get_device():
    """Get best available device (CUDA if available)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def generate_rectangle_mask_gpu(shape_data: Dict, width: int, height: int, device: torch.device) -> torch.Tensor:
    """Generate rectangle mask on GPU using vectorized operations"""
    x1 = shape_data['x']
    y1 = shape_data['y']
    x2 = x1 + shape_data['width']
    y2 = y1 + shape_data['height']

    # Create coordinate grids on GPU
    y_coords = torch.arange(height, device=device, dtype=torch.float32)
    x_coords = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Vectorized comparison
    mask = ((xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)).float()
    return mask


def generate_ellipse_mask_gpu(shape_data: Dict, width: int, height: int, device: torch.device) -> torch.Tensor:
    """Generate ellipse mask on GPU using ellipse equation"""
    cx = shape_data['cx']
    cy = shape_data['cy']
    rx = shape_data['rx']
    ry = shape_data['ry']

    # Avoid division by zero
    if rx < 1:
        rx = 1
    if ry < 1:
        ry = 1

    # Create coordinate grids on GPU
    y_coords = torch.arange(height, device=device, dtype=torch.float32)
    x_coords = torch.arange(width, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # Ellipse equation: ((x-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1
    ellipse_eq = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    mask = (ellipse_eq <= 1).float()
    return mask


def generate_polygon_mask_cv(shape_data: Dict, width: int, height: int) -> np.ndarray:
    """Generate polygon mask using OpenCV (CPU, but fast for complex shapes)"""
    points = shape_data['points']
    pts = np.array([[int(p['x']), int(p['y'])] for p in points], dtype=np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def generate_brush_mask_cv(shape_data: Dict, width: int, height: int) -> np.ndarray:
    """Generate brush stroke mask using OpenCV polylines"""
    points = shape_data['points']
    brush_size = shape_data.get('brushSize', 20)

    pts = np.array([[int(p['x']), int(p['y'])] for p in points], dtype=np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=brush_size)
    return mask


def generate_mask_gpu(shape_data: Dict, width: int, height: int, device: Optional[str] = None) -> np.ndarray:
    """
    Generate a binary mask from shape parameters.

    Args:
        shape_data: Dictionary with shape type and parameters
            - rectangle: {type, x, y, width, height}
            - ellipse: {type, cx, cy, rx, ry}
            - polygon: {type, points: [{x, y}, ...]}
            - brush: {type, points: [{x, y}, ...], brushSize}
        width: Video/mask width
        height: Video/mask height
        device: 'cuda' or 'cpu' (auto-detect if None)

    Returns:
        numpy array (H, W) with values 0 or 255
    """
    if device is None:
        dev = get_device()
    else:
        dev = torch.device(device)

    shape_type = shape_data.get('type', 'rectangle')

    if shape_type == 'rectangle':
        mask_tensor = generate_rectangle_mask_gpu(shape_data, width, height, dev)
        mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)

    elif shape_type == 'ellipse':
        mask_tensor = generate_ellipse_mask_gpu(shape_data, width, height, dev)
        mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)

    elif shape_type == 'polygon':
        # Use OpenCV for polygon (more efficient for complex shapes)
        mask_np = generate_polygon_mask_cv(shape_data, width, height)

    elif shape_type == 'brush':
        # Use OpenCV for brush strokes
        mask_np = generate_brush_mask_cv(shape_data, width, height)

    else:
        raise ValueError(f"Unknown shape type: {shape_type}")

    return mask_np


def generate_combined_mask(shapes: List[Dict], width: int, height: int, device: Optional[str] = None) -> np.ndarray:
    """
    Generate combined mask from multiple shapes (union).

    Args:
        shapes: List of shape dictionaries
        width: Mask width
        height: Mask height
        device: Device to use

    Returns:
        Combined mask (H, W) uint8
    """
    combined = np.zeros((height, width), dtype=np.uint8)

    for shape in shapes:
        mask = generate_mask_gpu(shape, width, height, device)
        combined = np.maximum(combined, mask)

    return combined


def generate_and_save_mask(shape_data: Union[Dict, List[Dict]], width: int, height: int,
                           output_path: str, device: Optional[str] = None) -> str:
    """
    Generate mask and save to file.

    Args:
        shape_data: Single shape dict or list of shapes
        width: Video width
        height: Video height
        output_path: Output PNG file path
        device: Device to use

    Returns:
        Output path
    """
    if isinstance(shape_data, list):
        mask = generate_combined_mask(shape_data, width, height, device)
    else:
        mask = generate_mask_gpu(shape_data, width, height, device)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    cv2.imwrite(output_path, mask)
    return output_path


def mask_to_base64(mask: np.ndarray) -> str:
    """Convert mask array to base64 PNG string"""
    _, buffer = cv2.imencode('.png', mask)
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_mask(b64_string: str) -> np.ndarray:
    """Convert base64 PNG string to mask array"""
    img_data = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    mask = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return mask


def dilate_mask(mask: np.ndarray, iterations: int = 5) -> np.ndarray:
    """Dilate mask by specified iterations"""
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask, kernel, iterations=iterations)


def replicate_mask_for_frames(mask: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """
    Create list of mask references for all frames (no copying - same mask).
    Memory efficient: all frames point to same numpy array.
    """
    return [mask] * num_frames


def main():
    parser = argparse.ArgumentParser(description='GPU-Accelerated Static Mask Generator')
    parser.add_argument('--shape', type=str, required=True,
                        help='JSON shape data (single shape or array of shapes)')
    parser.add_argument('--width', type=int, required=True,
                        help='Output mask width')
    parser.add_argument('--height', type=int, required=True,
                        help='Output mask height')
    parser.add_argument('--output', type=str, required=True,
                        help='Output PNG file path')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu, auto-detect if not specified)')
    parser.add_argument('--dilate', type=int, default=0,
                        help='Dilation iterations (default: 0)')

    args = parser.parse_args()

    # Parse shape data
    try:
        shape_data = json.loads(args.shape)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON shape data: {e}")
        return 1

    # Determine device
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"[MASK] Device: {device}")
    print(f"[MASK] Size: {args.width}x{args.height}")
    print(f"[MASK] Shape type: {shape_data.get('type', 'unknown') if isinstance(shape_data, dict) else 'multiple'}")

    try:
        # Generate mask
        output_path = generate_and_save_mask(shape_data, args.width, args.height, args.output, device)

        # Optional dilation
        if args.dilate > 0:
            mask = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE)
            mask = dilate_mask(mask, args.dilate)
            cv2.imwrite(output_path, mask)
            print(f"[MASK] Applied {args.dilate} dilation iterations")

        print(f"[OK] Mask saved to: {output_path}")
        return 0

    except Exception as e:
        print(f"[ERROR] Failed to generate mask: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
