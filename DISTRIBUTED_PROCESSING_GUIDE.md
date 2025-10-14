# Distributed Video Processing Guide

## Overview

The watermark removal system now supports **distributed processing** where multiple Celery workers across different computers collaborate on processing ONE video together. This dramatically speeds up processing by splitting the work across all available GPUs.

## How It Works

### Traditional (Single Machine):
```
Computer 1: Process entire video alone
  ├─ Segment 1 (GPU 0)
  ├─ Segment 2 (GPU 1)
  └─ Segment 3 (GPU 0) ... sequential within machine
```

### NEW Distributed (Multi-Machine):
```
Computer 1 & Computer 2 working on SAME video:
  ├─ Computer 1: Segment 1 (GPU 0) ┐
  ├─ Computer 2: Segment 2 (GPU 0) ├─ ALL PARALLEL
  ├─ Computer 1: Segment 3 (GPU 1) ┘
  └─ ... segments distributed across ALL workers
```

## Architecture

### Phase 1: Prepare Video (Master Worker)
**Task:** `prepare_video_task`
- Downloads video
- Runs YOLO detection on all frames
- Generates masks
- Detects segments based on watermark position
- Extracts frames to shared storage
- Returns segment metadata

### Phase 2: Process Segments (All Workers in Parallel)
**Task:** `process_segment_task` (runs once per segment)
- Each worker pulls one segment from the queue
- Downloads required frames and masks
- Crops to watermark region
- Runs ProPainter with RAFT and faster-propainter optimizations
- Merges cleaned region back to full frames
- Encodes segment video
- Uploads result to shared storage

**KEY**: This phase uses Celery's `group()` to run ALL segments in parallel across ALL available workers!

### Phase 3: Finalize Video (Master Worker)
**Task:** `finalize_video_task`
- Collects all segment videos
- Concatenates in correct order
- Merges audio from original
- Returns final result

### Coordination
Uses Celery's **chord pattern**:
```python
chord(
    group([process_segment_task for each segment]),  # Parallel
    finalize_video_task                              # Runs when all complete
)
```

## Setup Instructions

### 1. Configure Redis (Shared Broker)

All workers must connect to the **same Redis instance**:

```bash
# On the main server (e.g., Computer 1)
redis-server

# Get the IP address
ip addr show  # e.g., 192.168.1.100
```

### 2. Set Environment Variables

**On ALL computers:**

```bash
# Point to shared Redis broker
export REDIS_URL="redis://192.168.1.100:6379/0"

# Point to main API server (for file sharing)
export TUNNEL_URL="http://192.168.1.100:9000"

# Enable result upload
export UPLOAD_RESULT_BACK="1"
```

### 3. Start Flask API Server (Main Computer Only)

```bash
cd /app/waterz/WATERZ/web
python server_production.py
```

This serves:
- `/api/upload-segment` - Receives segment videos from workers
- `/temp/<path>` - Serves shared frames/masks to workers
- `/api/upload-result` - Receives final video

### 4. Start Celery Workers (ALL Computers)

**Computer 1:**
```bash
cd /app/waterz/WATERZ/web
celery -A server_production.celery worker --loglevel=info --concurrency=1 --hostname=worker1@%h
```

**Computer 2:**
```bash
cd /app/waterz/WATERZ/web
celery -A server_production.celery worker --loglevel=info --concurrency=1 --hostname=worker2@%h
```

**Important flags:**
- `--concurrency=1`: Process one video task at a time (but segments run in parallel)
- `--hostname=worker1@%h`: Unique name for each worker

### 5. Submit a Video for Distributed Processing

Use the NEW distributed task:

```python
from server_production import process_video_distributed_task

# Queue video for distributed processing
task = process_video_distributed_task.apply_async(args=['/path/to/video.mp4'])

# Check status
result = task.get()  # Blocks until complete
print(f"Result: {result['path']}")
```

## What You'll See in Logs

### Computer 1:
```
🚀 Starting DISTRIBUTED video processing: video.mp4
📋 Phase 1: Preparing video for distribution...
✅ Video prepared: 8 segments ready for distributed processing
🌐 Distributing segments across all available workers...
🔥 Phase 2: Processing 8 segments in parallel...
📊 Workflow submitted:
   - 8 segment tasks queued
   - Tasks will be picked up by ANY available worker
```

### Computer 2:
```
🎬 Worker processing segment 3/8: frames 120-240
   📐 Crop region: x=100, y=50, w=200, h=100
   ⬇️  Downloading frames 120-240...
   ✅ Downloaded 121 frames
   ✂️  Cropping frames...
   🎨 Running ProPainter...
   ✅ ProPainter complete for segment 3
   🔗 Merging cleaned region...
   🎞️  Encoding segment...
   ⬆️  Uploading segment 3 to API server...
✅ Segment 3/8 complete!
```

### Computer 1 (After all segments complete):
```
🎬 Finalizing video: video (8 segments)
📥 Collecting 8 segment videos...
✅ All 8 segments collected
🔗 Concatenating segments...
🎵 Merging audio...
✅ DISTRIBUTED processing complete!
```

## Performance Benefits

### Example: 5-minute video with 8 segments

**Single Machine (2 GPUs):**
- 8 segments × 30 seconds = 240 seconds = **4 minutes**

**Distributed (2 Machines × 2 GPUs = 4 GPUs):**
- 8 segments ÷ 4 workers × 30 seconds = **60 seconds**
- **4x faster!**

### Scaling

With N machines, processing time approaches:
```
Time ≈ (Total segments × Time per segment) / Total GPUs
```

## Troubleshooting

### Workers not picking up segments
**Check:** All workers connected to same Redis?
```bash
# On each worker
celery -A server_production.celery inspect active_queues
```

### Segments failing to download frames
**Check:** TUNNEL_URL set correctly on all workers?
```bash
echo $TUNNEL_URL
# Should point to main API server
```

### Final video missing segments
**Check:** Are segment uploads succeeding?
- Look for "✅ Segment X uploaded successfully" in worker logs
- Check `/results/` directory on main server

### Network firewall issues
**Open these ports:**
- Redis: 6379
- API Server: 9000

## Configuration Options

### Adjust Number of Segments
Edit `segment_detector.py`:
```python
# Minimum frames per segment
min_segment_length=10  # Smaller = more segments = more parallelism
```

### Adjust Segment Workers (Local GPUs)
**NOT NEEDED for distributed mode!**

The `SEGMENT_WORKERS` environment variable is for local multi-GPU processing within one machine. For distributed processing, Celery automatically distributes segments across ALL workers.

## API Changes

### Old Way (Single Machine):
```python
process_video_task.apply_async(args=[video_path])
```

### NEW Way (Distributed):
```python
process_video_distributed_task.apply_async(args=[video_path])
```

**Backward Compatible:** Old `process_video_task` still works for single-machine processing!

## Technical Details

### Task Names
- `watermark.prepare_video` - Preparation phase
- `watermark.process_segment` - Segment processing (parallel)
- `watermark.finalize_video` - Finalization phase
- `watermark.remove_video_distributed` - Main orchestrator

### Celery Configuration
```python
worker_prefetch_multiplier=1  # One task at a time
task_acks_late=True            # Retry if worker dies
```

### File Sharing
- **Masks:** Shared via `/temp/propainter_masks/{video_id}/`
- **Frames:** Shared via `/temp/{video_id}_originals/`
- **Segments:** Uploaded via `/api/upload-segment`

## Summary

**Before:** One computer, one video, sequential segments
**After:** All computers collaborate, one video, parallel segments

**Result:** Massive speedup by combining all available GPU power!

🚀 **You can now process videos much faster by simply adding more Celery workers across different computers!**
