"""
Watermark position segmentation for smart cropping optimization.

Tracks watermark position changes and groups frames into static segments.
"""

from typing import List, Tuple, Optional
import numpy as np


def bbox_distance(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    """
    Calculate distance between two bounding boxes (center point distance).

    Args:
        bbox1: (x1, y1, x2, y2)
        bbox2: (x1, y1, x2, y2)

    Returns:
        Euclidean distance between bbox centers
    """
    center1_x = (bbox1[0] + bbox1[2]) / 2
    center1_y = (bbox1[1] + bbox1[3]) / 2
    center2_x = (bbox2[0] + bbox2[2]) / 2
    center2_y = (bbox2[1] + bbox2[3]) / 2

    dx = center2_x - center1_x
    dy = center2_y - center1_y

    return np.sqrt(dx * dx + dy * dy)


def bboxes_similar(
    bbox1: Tuple[int, int, int, int],
    bbox2: Tuple[int, int, int, int],
    position_tolerance: int = 5,
    size_tolerance: float = 0.1
) -> bool:
    """
    Check if two bounding boxes are similar (same position and size).

    Args:
        bbox1: (x1, y1, x2, y2)
        bbox2: (x1, y1, x2, y2)
        position_tolerance: Max pixel difference for position (default 5px)
        size_tolerance: Max fractional difference for size (default 10%)

    Returns:
        True if bboxes are considered the same watermark
    """
    # Check position (center distance)
    distance = bbox_distance(bbox1, bbox2)
    if distance > position_tolerance:
        return False

    # Check size
    w1 = bbox1[2] - bbox1[0]
    h1 = bbox1[3] - bbox1[1]
    w2 = bbox2[2] - bbox2[0]
    h2 = bbox2[3] - bbox2[1]

    size1 = w1 * h1
    size2 = w2 * h2

    if size1 == 0 or size2 == 0:
        return size1 == size2

    size_ratio = max(size1, size2) / min(size1, size2)
    if size_ratio > (1.0 + size_tolerance):
        return False

    return True


def average_bbox(bboxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    """
    Calculate average bounding box from a list of similar bboxes.

    Args:
        bboxes: List of (x1, y1, x2, y2) tuples

    Returns:
        Average bbox (x1, y1, x2, y2) as integers
    """
    if not bboxes:
        return (0, 0, 0, 0)

    x1_avg = int(np.mean([b[0] for b in bboxes]))
    y1_avg = int(np.mean([b[1] for b in bboxes]))
    x2_avg = int(np.mean([b[2] for b in bboxes]))
    y2_avg = int(np.mean([b[3] for b in bboxes]))

    return (x1_avg, y1_avg, x2_avg, y2_avg)


def detect_segments(
    detections_per_frame: List[Optional[Tuple[int, int, int, int]]],
    position_tolerance: int = 5,
    min_segment_length: int = 10
) -> List[Tuple[int, int, Tuple[int, int, int, int]]]:
    """
    Detect segments where watermark stays in same position.

    Args:
        detections_per_frame: List of bbox per frame (None if no detection)
        position_tolerance: Max pixel movement to consider "static" (default 5px)
        min_segment_length: Minimum frames for a valid segment (default 10)

    Returns:
        List of (start_frame, end_frame, average_bbox) tuples
        Example: [(0, 450, (100,50,200,100)), (451, 720, (105,55,205,105))]
    """
    if not detections_per_frame:
        return []

    segments = []
    current_segment_start = None
    current_segment_bboxes = []

    for frame_idx, bbox in enumerate(detections_per_frame):
        if bbox is None:
            # No detection - end current segment if exists
            if current_segment_start is not None and len(current_segment_bboxes) >= min_segment_length:
                avg_bbox = average_bbox(current_segment_bboxes)
                segments.append((current_segment_start, frame_idx - 1, avg_bbox))
            current_segment_start = None
            current_segment_bboxes = []
            continue

        # We have a detection
        if current_segment_start is None:
            # Start new segment
            current_segment_start = frame_idx
            current_segment_bboxes = [bbox]
        else:
            # Check if bbox is similar to segment
            avg_bbox_so_far = average_bbox(current_segment_bboxes)
            if bboxes_similar(bbox, avg_bbox_so_far, position_tolerance=position_tolerance):
                # Continue current segment
                current_segment_bboxes.append(bbox)
            else:
                # Position changed - end current segment and start new one
                if len(current_segment_bboxes) >= min_segment_length:
                    avg_bbox = average_bbox(current_segment_bboxes)
                    segments.append((current_segment_start, frame_idx - 1, avg_bbox))

                current_segment_start = frame_idx
                current_segment_bboxes = [bbox]

    # Handle final segment
    if current_segment_start is not None and len(current_segment_bboxes) >= min_segment_length:
        avg_bbox = average_bbox(current_segment_bboxes)
        segments.append((current_segment_start, len(detections_per_frame) - 1, avg_bbox))

    return segments


def merge_adjacent_segments(
    segments: List[Tuple[int, int, Tuple[int, int, int, int]]],
    position_tolerance: int = 5,
    max_gap: int = 30
) -> List[Tuple[int, int, Tuple[int, int, int, int]]]:
    """
    Merge adjacent segments with similar bbox positions.

    Useful for handling brief detection gaps or slight jitter.

    Args:
        segments: List of (start, end, bbox) tuples
        position_tolerance: Max pixel difference to consider mergeable
        max_gap: Max frame gap between segments to consider merging

    Returns:
        Merged segments list
    """
    if len(segments) <= 1:
        return segments

    merged = []
    current = segments[0]

    for next_seg in segments[1:]:
        current_start, current_end, current_bbox = current
        next_start, next_end, next_bbox = next_seg

        gap = next_start - current_end
        similar = bboxes_similar(current_bbox, next_bbox, position_tolerance=position_tolerance)

        if gap <= max_gap and similar:
            # Merge: extend current segment to include next
            all_bboxes = [current_bbox] * (current_end - current_start + 1)
            all_bboxes.extend([next_bbox] * (next_end - next_start + 1))
            merged_bbox = average_bbox(all_bboxes)
            current = (current_start, next_end, merged_bbox)
        else:
            # Can't merge - save current and move to next
            merged.append(current)
            current = next_seg

    # Add final segment
    merged.append(current)

    return merged


def should_use_cropping(
    segments: List[Tuple[int, int, Tuple[int, int, int, int]]],
    frame_width: int,
    frame_height: int,
    min_speedup: float = 5.0
) -> bool:
    """
    Determine if cropping optimization is worthwhile for this video.

    Args:
        segments: Detected segments
        frame_width: Full frame width
        frame_height: Full frame height
        min_speedup: Minimum estimated speedup to enable cropping (default 5x)

    Returns:
        True if cropping should be used
    """
    if not segments:
        return False

    # Calculate average watermark size across segments
    total_pixels = 0
    total_frames = 0

    for start, end, bbox in segments:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        crop_pixels = w * h
        frame_count = end - start + 1

        total_pixels += crop_pixels * frame_count
        total_frames += frame_count

    if total_frames == 0:
        return False

    avg_crop_pixels = total_pixels / total_frames
    full_frame_pixels = frame_width * frame_height

    # Estimate speedup (with overhead factor)
    estimated_speedup = (full_frame_pixels / avg_crop_pixels) * 0.7

    return estimated_speedup >= min_speedup


# Test/example
if __name__ == "__main__":
    # Example: Video with watermark that moves twice
    detections = []

    # Frames 0-99: Watermark at (100, 50, 200, 100)
    for i in range(100):
        detections.append((100, 50, 200, 100))

    # Frames 100-199: Watermark moves to (110, 55, 210, 105)
    for i in range(100):
        detections.append((110, 55, 210, 105))

    # Frames 200-250: No watermark
    for i in range(51):
        detections.append(None)

    # Frames 251-349: Watermark back at (100, 50, 200, 100)
    for i in range(99):
        detections.append((100, 50, 200, 100))

    print("Detecting segments...")
    segments = detect_segments(detections, position_tolerance=5, min_segment_length=10)

    print(f"\nFound {len(segments)} segments:")
    for start, end, bbox in segments:
        duration = end - start + 1
        print(f"  Frames {start}-{end} ({duration} frames): bbox={bbox}")

    print("\nMerging adjacent segments...")
    merged = merge_adjacent_segments(segments, position_tolerance=5, max_gap=30)

    print(f"\nAfter merging: {len(merged)} segments:")
    for start, end, bbox in merged:
        duration = end - start + 1
        print(f"  Frames {start}-{end} ({duration} frames): bbox={bbox}")

    print("\nShould use cropping?")
    use_crop = should_use_cropping(merged, 1920, 1080, min_speedup=5.0)
    print(f"  {use_crop} (estimated speedup > 5x)")
