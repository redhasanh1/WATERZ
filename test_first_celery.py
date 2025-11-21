"""
Test first.mp4 with Celery server
Compares quality with local RUN_SAM2_LOCAL.py
"""
from server_production import process_video_task
import time

video_path = "D:\\watermarkz\\videostotrain\\first.mp4"
print(f"=" * 60)
print(f"CELERY SERVER TEST - Processing first.mp4")
print(f"=" * 60)
print(f"Video: {video_path}")
print()

# Submit task to Celery
print("Submitting to Celery worker...")
result = process_video_task.apply_async(args=[video_path])
task_id = result.id
print(f"Task ID: {task_id}")
print()

# Wait for completion
print("Waiting for processing...")
start_time = time.time()

try:
    final_result = result.get(timeout=600)  # 10 minute timeout
    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("✅ SUCCESS!")
    print("=" * 60)
    print(f"Time: {elapsed:.1f} seconds")
    print(f"Result: {final_result}")
    print()
    print("Check output in: D:\\watermarkz\\results\\")

except Exception as e:
    print()
    print("=" * 60)
    print("❌ FAILED")
    print("=" * 60)
    print(f"Error: {e}")
