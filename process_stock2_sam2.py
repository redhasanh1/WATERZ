"""
Quick test script to process stock2.mp4 with SAM2 masks
Uses existing masks from temp_sam2_masks folder
"""

import os
import time
import requests
import json

# Configuration
VIDEO_PATH = r"D:\watermarkz\videostotrain\stock2.mp4"
MASKS_FOLDER = r"D:\watermarkz\temp_sam2_masks"
VIDEO_ID = "stock2"

# API endpoint (Flask server on port 5001)
API_URL = "http://localhost:5001"

def main():
    print("="*80)
    print("SAM2 WATERMARK REMOVAL - TEST SCRIPT")
    print("="*80)
    print()

    # Check inputs
    print(f"[CHECK] Video: {VIDEO_PATH}")
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video not found: {VIDEO_PATH}")
        return
    print(f"[OK] Video exists")

    print(f"[CHECK] Masks: {MASKS_FOLDER}")
    if not os.path.exists(MASKS_FOLDER):
        print(f"[ERROR] Masks folder not found: {MASKS_FOLDER}")
        return

    mask_files = [f for f in os.listdir(MASKS_FOLDER) if f.endswith('.png')]
    print(f"[OK] Found {len(mask_files)} mask files")
    print()

    # Submit task
    print("[SUBMIT] Submitting SAM2 task to Celery...")
    try:
        response = requests.post(
            f"{API_URL}/api/process_sam2",
            json={
                'video_path': VIDEO_PATH,
                'masks_folder': MASKS_FOLDER,
                'video_id': VIDEO_ID
            }
        )

        if response.status_code != 200:
            print(f"[ERROR] API returned status {response.status_code}")
            print(f"[ERROR] Response: {response.text}")
            return

        result = response.json()
        task_id = result.get('task_id')

        print(f"[OK] Task queued: {task_id}")
        print(f"[OK] Status: {result.get('status')}")
        print()

    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Could not connect to API at {API_URL}")
        print(f"[ERROR] Make sure Flask server is running:")
        print(f"[ERROR]   python server_production2.py")
        return

    # Poll for completion
    print("[POLL] Waiting for task to complete...")
    print()

    last_status = None
    while True:
        try:
            response = requests.get(f"{API_URL}/api/task/{task_id}")
            data = response.json()

            state = data.get('state')
            current_status = data.get('status', '')
            current_progress = data.get('current', 0)

            # Only print if status changed
            if current_status != last_status:
                print(f"[{state}] {current_status} ({current_progress}%)")
                last_status = current_status

            if state == 'SUCCESS':
                result = data.get('result')
                print()
                print("="*80)
                print("✅ SAM2 PROCESSING COMPLETE!")
                print("="*80)
                print(f"Video ID: {result.get('video_id')}")
                print(f"Total Frames: {result.get('total_frames')}")
                print(f"Resolution: {result.get('width')}x{result.get('height')}")
                print(f"FPS: {result.get('fps')}")
                print()
                print(f"Output: {result.get('output_path')}")
                print("="*80)
                break

            elif state == 'FAILURE':
                print()
                print("="*80)
                print("❌ TASK FAILED")
                print("="*80)
                print(f"Error: {data.get('status')}")
                print("="*80)
                break

            # Wait before next poll
            time.sleep(2)

        except KeyboardInterrupt:
            print()
            print("[CANCELLED] Stopping poll (task still running in background)")
            break
        except Exception as e:
            print(f"[ERROR] Poll failed: {e}")
            time.sleep(2)

if __name__ == "__main__":
    main()
