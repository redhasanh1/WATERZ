# Cloud Worker Deployment Fixes

## Changes Made

### 1. Enable Parallel Segmentation (Lines 34-36)
**Problem:** Videos were being processed as 1 segment, preventing multi-worker parallelism.

**Fix:** Added forced segmentation configuration:
```python
os.environ.setdefault('MIN_SEGMENTS', '4')  # Split videos into at least 4 segments
os.environ.setdefault('MIN_CHUNK_FRAMES', '30')  # Minimum frames per segment
```

**Result:** Videos will now be split into 4+ segments for parallel processing across workers.

---

### 2. Enhanced ProPainter Error Logging (Lines 1219-1225, 1244-1246)
**Problem:** ProPainter failures were silent, making debugging impossible.

**Fix:** Added verbose logging and full traceback:
```python
print(f"   📦 Importing ProPainter from: {faster_propainter_path}")
from watermark import pipeline as faster_propainter_pipeline
print(f"   ✅ ProPainter import successful")
print(f"   🔧 GPU available: {use_fp16}, Running ProPainter pipeline...")
```

**Result:** Will see exactly where ProPainter fails (import, GPU check, or pipeline execution).

---

## Deployment Steps for Cloud Worker

### Step 1: Copy Updated File
```bash
# On cloud worker at /app/
cp /app/server_production.py /app/server_production.py.backup
# Then upload the new CLOUD_WORKER_WITH_IMAGES_2.txt as server_production.py
```

### Step 2: Verify faster-propainter-main Structure
```bash
# Make sure these exist:
ls -la /app/faster-propainter-main/model/modules/
ls -la /app/faster-propainter-main/core/
ls -la /app/faster-propainter-main/utils/
ls -la /app/faster-propainter-main/RAFT/

# If missing, run the setup command:
cd /app && \
rm -rf ProPainter && \
git clone https://github.com/sczhou/ProPainter.git && \
cp -r ProPainter/model ProPainter/core ProPainter/utils ProPainter/RAFT /app/faster-propainter-main/ && \
echo "✅ Directories copied to faster-propainter-main"
```

### Step 3: Test ProPainter Import
```bash
python3 -c "import sys; sys.path.insert(0, '/app/faster-propainter-main'); from watermark import pipeline; print('✅ ProPainter ready')"
```

### Step 4: Restart Workers
```bash
# Kill existing workers
pkill -f celery

# Start workers (adjust GPU assignments as needed)
CUDA_VISIBLE_DEVICES=0 celery -A server_production worker --loglevel=info --concurrency=1 --hostname=worker1@%h &
CUDA_VISIBLE_DEVICES=1 celery -A server_production worker --loglevel=info --concurrency=1 --hostname=worker2@%h &
CUDA_VISIBLE_DEVICES=2 celery -A server_production worker --loglevel=info --concurrency=1 --hostname=worker3@%h &
CUDA_VISIBLE_DEVICES=3 celery -A server_production worker --loglevel=info --concurrency=1 --hostname=worker4@%h &
```

---

## Expected Behavior After Fix

### Before:
```
📊 Detected 1 segments for distributed processing
   Segment 1: frames 15-91 (77 frames)
🔥 Dispatching 1 segment tasks manually across all workers...
   ✅ Segment 1 task queued
```
**Only 1 worker processes the entire video.**

### After:
```
📊 Detected 1 segments for distributed processing
   Segment 1: frames 15-91 (77 frames)
🔀 Force-splitting into 4 segments for parallel distribution
   Segment 1: frames 15-33 (19 frames)
   Segment 2: frames 34-52 (19 frames)
   Segment 3: frames 53-71 (19 frames)
   Segment 4: frames 72-91 (20 frames)
🔥 Dispatching 4 segment tasks manually across all workers...
   ✅ Segment 1 task queued
   ✅ Segment 2 task queued
   ✅ Segment 3 task queued
   ✅ Segment 4 task queued
```
**All 4 workers process segments in parallel, 4x faster!**

---

## Troubleshooting

### If watermarks still visible:
1. Check ProPainter logs for import errors
2. Verify model weights exist: `ls -lh /app/faster-propainter-main/model/*.pth`
3. Look for "ProPainter import successful" in logs
4. Check GPU availability: logs should show "GPU available: True"

### If still single segment:
1. Verify MIN_SEGMENTS is set: `grep MIN_SEGMENTS /app/server_production.py`
2. Check if video is too short (< 60 frames)
3. Look for "Force-splitting" message in logs

### If workers not picking up tasks:
1. Verify Redis connection
2. Check worker names are unique (worker1, worker2, etc.)
3. Ensure workers are running on different GPUs (CUDA_VISIBLE_DEVICES)
