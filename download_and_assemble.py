"""
Download/collect all segment results and assemble into final video on Windows

NEW BATCH ARCHITECTURE:
- Results are saved LOCALLY in Docker (no B2 uploads during processing)
- Docker maps /app/results to a Windows directory
- This script collects from local paths and assembles
"""
import os
import sys
import time
import zipfile
import requests
import cv2
import json
import subprocess
import shutil

# Redis connection
try:
    import redis
except ImportError:
    print("Installing redis...")
    subprocess.run([sys.executable, "-m", "pip", "install", "redis", "-q"])
    import redis

REDIS_URL = "redis://default:bwQmxUCQEXUlYTWACmPbbkpnHPVpoiIa@tramway.proxy.rlwy.net:48930"
CLOUDFLARE_BASE = "https://markz.humblewoslayer.workers.dev"
OUTPUT_DIR = r"D:\watermarkz\output"

# Docker results directory mapping:
# Docker: /app/results -> Windows: D:\watermarkz\docker_results
DOCKER_RESULTS_DIR = r"D:\watermarkz\docker_results"


def wait_for_batch(video_id, timeout=600):
    """Wait for batch mode to complete (single job, ALL frames)"""
    r = redis.from_url(REDIS_URL)

    print(f"[BATCH MODE] Waiting for batch processing to complete...")
    start = time.time()

    while time.time() - start < timeout:
        status = r.get(f'batch_status:{video_id}')
        if status:
            status = status.decode() if isinstance(status, bytes) else status
            if status == 'completed':
                result_path = r.get(f'batch_result:{video_id}')
                if result_path:
                    result_path = result_path.decode() if isinstance(result_path, bytes) else result_path
                    print(f"\n[OK] Batch processing complete!")
                    return {0: result_path}
            elif status == 'failed':
                print(f"\n[ERROR] Batch processing failed!")
                return {}

        print(f"  Batch status: {status or 'waiting'}...", end='\r')
        time.sleep(2)

    print(f"\n[TIMEOUT] Batch not complete after {timeout}s")
    return {}


def wait_for_segments(video_id, total_segments=None, timeout=600):
    """Wait for all segments to complete"""
    r = redis.from_url(REDIS_URL)

    # Check if this is batch mode (total_segments = 1 with batch_status key)
    batch_status = r.get(f'batch_status:{video_id}')
    if batch_status:
        return wait_for_batch(video_id, timeout)

    # Get total_segments from Redis if not provided
    if total_segments is None:
        ts = r.get(f'total_segments:{video_id}')
        if ts:
            total_segments = int(ts.decode() if isinstance(ts, bytes) else ts)
        else:
            print(f"[ERROR] Could not find total_segments for {video_id}")
            return {}

    print(f"Waiting for {total_segments} segments to complete...")
    start = time.time()

    while time.time() - start < timeout:
        completed = 0
        results = {}

        for i in range(total_segments):
            status = r.get(f'segment_status:{video_id}:{i}')
            if status:
                status = status.decode() if isinstance(status, bytes) else status
                if status == 'completed':
                    completed += 1
                    # Get result path (local path, not URL!)
                    path = r.hget(f'segment_results:{video_id}', i)
                    if path:
                        results[i] = path.decode() if isinstance(path, bytes) else path

        print(f"  Progress: {completed}/{total_segments} segments complete", end='\r')

        if completed == total_segments:
            print(f"\n[OK] All {total_segments} segments complete!")
            return results

        time.sleep(2)

    print(f"\n[TIMEOUT] Only {completed}/{total_segments} complete after {timeout}s")
    return results


def collect_local_segment(docker_path, output_dir):
    """
    Collect frames from local Docker results.
    Converts Docker path (/app/results/...) to Windows path.
    """
    # Convert Docker path to Windows path
    # Docker: /app/results/video_id_segments/segment_X
    # Windows: D:\watermarkz\docker_results\video_id_segments\segment_X
    if docker_path.startswith('/app/results/'):
        relative_path = docker_path[len('/app/results/'):]
        windows_path = os.path.join(DOCKER_RESULTS_DIR, relative_path)
    else:
        # Already a Windows path
        windows_path = docker_path

    print(f"  Collecting from: {windows_path}")

    if not os.path.exists(windows_path):
        print(f"  [ERROR] Path does not exist: {windows_path}")
        return []

    # Get all frame files
    frames = sorted([f for f in os.listdir(windows_path) if f.endswith('.png')])
    return [os.path.join(windows_path, f) for f in frames]


def download_segment(url, output_dir):
    """Download and extract segment zip (legacy B2 method)"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"  Downloading: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    zip_path = os.path.join(output_dir, "segment.zip")
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=65536):
            if chunk:
                f.write(chunk)

    # Extract
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)

    os.remove(zip_path)

    # Return list of frame files
    frames = sorted([f for f in os.listdir(output_dir) if f.endswith('.png')])
    return [os.path.join(output_dir, f) for f in frames]

def assemble_video(all_frames, output_path, fps=30):
    """Create video from frames using cv2.VideoWriter"""
    if not all_frames:
        print("[ERROR] No frames to assemble!")
        return

    # Read first frame to get dimensions
    first = cv2.imread(all_frames[0])
    height, width = first.shape[:2]

    print(f"Creating video: {width}x{height} @ {fps}fps, {len(all_frames)} frames")

    # Use cv2.VideoWriter with mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i, frame_path in enumerate(all_frames):
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)
        if (i + 1) % 50 == 0:
            print(f"  Written {i + 1}/{len(all_frames)} frames...")

    out.release()

    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"[OK] Video saved: {output_path} ({size_mb:.2f} MB)")
        return output_path
    else:
        print("[ERROR] Video creation failed!")
        return None

def main():
    """
    NEW BATCH ARCHITECTURE:
    - Wait for all segments to complete
    - Collect frames from local Docker results (not B2)
    - Assemble video
    - Play on Windows
    """
    # Get video_id from command line or use latest from Redis
    if len(sys.argv) > 1:
        video_id = sys.argv[1]
        total_segments = int(sys.argv[2]) if len(sys.argv) > 2 else None
    else:
        # Find latest video_id in Redis
        r = redis.from_url(REDIS_URL)
        # For now, use a default
        video_id = input("Enter video_id (or press Enter for last test): ").strip()
        if not video_id:
            video_id = "test"  # Will be filled in by test script
        total_segments = None

    print(f"=" * 60)
    print(f"BATCH ASSEMBLY - Video: {video_id}")
    print(f"=" * 60)

    # Wait for all segments to complete
    results = wait_for_segments(video_id, total_segments)

    if not results:
        print(f"[ERROR] No segments found for video {video_id}")
        return

    total_found = len(results)
    print(f"Found {total_found} completed segments")

    # Collect frames from LOCAL paths (Docker results)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_frames = []

    for seg_idx in sorted(results.keys()):
        path_or_url = results[seg_idx]
        print(f"\nSegment {seg_idx}:")

        # Check if it's a local path or URL
        if path_or_url.startswith('http'):
            # Legacy: download from B2
            seg_dir = os.path.join(OUTPUT_DIR, f"segment_{seg_idx}")
            frames = download_segment(path_or_url, seg_dir)
        else:
            # NEW: collect from local Docker results
            frames = collect_local_segment(path_or_url, OUTPUT_DIR)

        all_frames.extend(frames)
        print(f"  Got {len(frames)} frames")

    print(f"\nTotal frames: {len(all_frames)}")

    if not all_frames:
        print("[ERROR] No frames collected!")
        return

    # Assemble final video
    output_path = os.path.join(OUTPUT_DIR, f"{video_id}_final.mp4")
    video_path = assemble_video(all_frames, output_path, fps=30)

    if video_path:
        print(f"\n" + "=" * 60)
        print(f"DONE! Output video: {video_path}")
        print(f"=" * 60)

        # Open in default player
        os.startfile(video_path)
    else:
        print("[ERROR] Video assembly failed!")


if __name__ == "__main__":
    main()
