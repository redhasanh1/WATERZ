# DISTRIBUTED PROCESSING FIX - Segment Tasks Not Being Picked Up

## 🚨 CRITICAL BUG IDENTIFIED

**Date:** 2025-10-18
**Issue:** Worker 2 successfully queued 2 segment tasks, but only Worker 1 picked up Segment 1. Segment 2 was never processed, causing the job to hang forever.

---

## 📊 Evidence from Logs

### Worker 2 (Prepare Task):
```
[18:15:12] ✅ Segment 1 task queued: 7489e325-9b09-4add-b3d0-8b560c9a2a76
[18:15:12] ✅ Segment 2 task queued: f1cb2a81-c406-4c80-9d38-fbc2f0fe664b
[18:15:12] ✅ All 2 segment tasks dispatched to queue!
[18:15:12] Task watermark.prepare_video succeeded
```
✅ Both segments dispatched to Redis

### Worker 1 (Segment Processing):
```
[18:15:16] Task watermark.process_segment[7489e325...] received
[18:15:18] 🎬 Worker processing segment 1/2: frames 0-65
[18:18:09] ✅ Segment 1/2 complete!
[18:18:09] 📊 Tracking: 1/2 segments complete
```
✅ Segment 1 processed successfully

### ❌ THE PROBLEM:
**Segment 2 (task f1cb2a81...) was NEVER picked up by any worker!**

Result: System waiting forever for 2/2 segments to complete (stuck at 1/2)

---

## 🔍 ROOT CAUSE ANALYSIS

### Issue #1: Solo Pool Mode
```bash
celery -A server_production.celery worker --pool=solo
```

**What `--pool=solo` does:**
- Uses a single-threaded event loop
- Can only process ONE task at a time
- Worker blocks completely while processing
- Cannot prefetch or queue additional tasks

**Why this broke distributed processing:**
1. Worker 2 finished `prepare_video_task` at 18:15:12
2. Both segment tasks queued in Redis
3. Worker 1 picked up Segment 1 at 18:15:16
4. **Worker 2 should have picked up Segment 2, but solo pool prevented it**
5. Worker 2 became "stuck" or unavailable after prepare task
6. Segment 2 remains in queue forever

---

### Issue #2: Worker Availability After Prepare Task

Solo pool workers can become unavailable after long-running tasks. The prepare task:
- Downloaded video (2s)
- Ran YOLO detection on 92 frames (16s)
- Extracted frames with ffmpeg (13s)
- Total: **35 seconds** of blocking execution

After this, Worker 2 should have been available, but logs show it never picked up Segment 2.

---

### Issue #3: Heartbeat Issues

```
[18:18:10] missed heartbeat from worker2@1ee93496...
[18:18:10] Substantial drift from worker2 may mean clocks are out of sync. Current drift is 178 seconds.
```

**Clock drift of 178 seconds** suggests:
- Worker 2 became unresponsive
- Celery couldn't reliably route tasks to it
- Solo pool may have caused deadlock or event loop hang

---

## ✅ THE SOLUTION

### Change Pool Mode from Solo to Threads

**Instead of:**
```bash
celery -A server_production.celery worker --pool=solo
```

**Use:**
```bash
celery -A server_production.celery worker --pool=threads --concurrency=1
```

### Why Threads Pool?

| Pool Type | Concurrency | Task Pickup | GPU Safety | Best For |
|-----------|-------------|-------------|------------|----------|
| **solo** | 1 (single event loop) | ❌ Can block | ✅ Yes | Development/debugging only |
| **threads** | Configurable | ✅ Non-blocking | ✅ Yes (with CUDA_VISIBLE_DEVICES) | **I/O-bound + GPU tasks** |
| **prefork** | Multiple processes | ✅ Non-blocking | ⚠️ Needs careful GPU management | CPU-bound tasks |

**Threads pool advantages:**
- Non-blocking task pickup (can grab new tasks while processing)
- Works with `concurrency=1` for single GPU
- Lower memory overhead than prefork
- Compatible with CUDA GPU assignment

---

## 🚀 CORRECTED WORKER STARTUP COMMANDS

### Worker 1 (Cloud GPU 0):
```bash
#!/bin/bash
export SEGMENT_WORKERS=1
export CUDA_VISIBLE_DEVICES=0
export MIN_SEGMENTS=2
export MIN_CHUNK_FRAMES=60
export REDIS_URL="redis://:watermarkz_secure_2024@8.tcp.ngrok.io:17609/0"

celery -A server_production.celery worker \
  --loglevel=info \
  --pool=threads \
  --concurrency=1 \
  -n worker1@%h
```

### Worker 2 (Cloud GPU 1):
```bash
#!/bin/bash
export SEGMENT_WORKERS=1
export CUDA_VISIBLE_DEVICES=1
export MIN_SEGMENTS=2
export MIN_CHUNK_FRAMES=60
export REDIS_URL="redis://:watermarkz_secure_2024@8.tcp.ngrok.io:17609/0"

celery -A server_production.celery worker \
  --loglevel=info \
  --pool=threads \
  --concurrency=1 \
  -n worker2@%h
```

**Key changes:**
- `--pool=solo` → `--pool=threads`
- `--concurrency=1` → Ensures each worker processes one segment at a time (GPU limitation)
- `-n worker1@%h` / `-n worker2@%h` → Unique worker names for tracking

---

## 🧪 VERIFICATION STEPS

After restarting workers with threads pool, verify distributed processing works:

### 1. Check Both Workers Are Registered
```bash
celery -A server_production.celery inspect registered
```

**Expected output:**
```
worker1@<hostname>:
  - watermark.process_segment
  - watermark.prepare_video
  - watermark.finalize_video

worker2@<hostname>:
  - watermark.process_segment
  - watermark.prepare_video
  - watermark.finalize_video
```

### 2. Check Active Tasks During Processing
```bash
celery -A server_production.celery inspect active
```

**Expected output (during segment processing):**
```
worker1@<hostname>:
  - watermark.process_segment[task-id-1] (segment 0)

worker2@<hostname>:
  - watermark.process_segment[task-id-2] (segment 1)
```
✅ Both workers processing different segments simultaneously!

### 3. Monitor Worker Logs for Parallel Execution

**Expected logs:**

Worker 1:
```
[18:20:15] Task watermark.process_segment[abc123...] received
[18:20:16] 🎬 Worker processing segment 1/2: frames 0-65
```

Worker 2:
```
[18:20:15] Task watermark.process_segment[def456...] received
[18:20:16] 🎬 Worker processing segment 2/2: frames 66-91
```

**Timing should overlap** → Both workers running ProPainter at the same time!

### 4. Check Finalize Task Triggers

**Expected logs:**
```
Worker 1:
[18:23:00] ✅ Segment 1/2 complete!
[18:23:00] 📊 Tracking: 1/2 segments complete

Worker 2:
[18:23:05] ✅ Segment 2/2 complete!
[18:23:05] 📊 Tracking: 2/2 segments complete
[18:23:05] 🎉 All segments complete! Triggering finalize...
[18:23:05] ✅ Finalize task triggered!
```

---

## 📈 EXPECTED PERFORMANCE IMPROVEMENT

### Before Fix (Solo Pool):
```
Video: 92 frames, 2 segments

Worker 2: [Prepare 35s] → [IDLE - stuck]
Worker 1:                  [Seg 1: 172s] → [waiting for Seg 2 forever...]

Total: STUCK (never completes)
```

### After Fix (Threads Pool):
```
Video: 92 frames, 2 segments

Worker 2: [Prepare 35s] → [Seg 2: ~90s*]
Worker 1:                  [Seg 1: 172s]
                                         ↓
                                    [Finalize 30s]

Total: ~237s (Prepare + max(Seg1, Seg2) + Finalize)
* Segment 2 is smaller (23 frames vs 66), should finish faster
```

**Speedup:** ~2x faster than sequential processing!

---

## 🔄 RECOVERY FROM STUCK STATE

If your system is currently stuck with segments in queue:

### 1. Purge All Pending Tasks
```bash
celery -A server_production.celery purge

# Confirm: y
```

### 2. Kill All Workers
```bash
# Press Ctrl+C in each worker terminal
# Or forcefully:
pkill -f "celery.*server_production"
```

### 3. Clear Redis Queue Manually (if needed)
```bash
redis-cli -h 8.tcp.ngrok.io -p 17609 -a watermarkz_secure_2024

# Check queue length
LLEN celery

# Purge queue
DEL celery

# Clear all task metadata
KEYS celery-task-meta-*
EVAL "return redis.call('del', unpack(redis.call('keys', ARGV[1])))" 0 celery-task-meta-*
```

### 4. Restart Workers with Threads Pool
```bash
# Terminal 1
bash start_worker1.sh

# Terminal 2
bash start_worker2.sh
```

### 5. Re-upload Video for Processing
Upload a new video through the API to test the fix.

---

## 🎯 MONITORING DISTRIBUTED PROCESSING

### Real-time Worker Stats
```bash
# Check what workers are doing
watch -n 2 "celery -A server_production.celery inspect active"
```

### Redis Queue Length
```bash
# Monitor queue depth
redis-cli -h 8.tcp.ngrok.io -p 17609 -a watermarkz_secure_2024 LLEN celery
```

### Task Success Rate
```bash
# Check completed tasks
celery -A server_production.celery inspect stats
```

---

## 🚨 WARNING SIGNS TO WATCH FOR

After implementing the fix, monitor for these issues:

### 1. Workers Missing Heartbeats
```
missed heartbeat from worker2@...
```
**Cause:** Network issues, clock drift, or worker overload
**Fix:** Increase broker_heartbeat, sync system clocks

### 2. Tasks Not Being Picked Up
```
✅ Segment 1 task queued: abc123
✅ Segment 2 task queued: def456
[long pause, no "received" messages]
```
**Cause:** Workers not registered, routing issues
**Fix:** Check `celery inspect registered`, verify queue names

### 3. One Worker Processing All Segments
```
Worker 1: Received segment 1
Worker 1: Received segment 2  ← Should go to Worker 2!
```
**Cause:** Worker 2 not available, prefetch settings
**Fix:** Set `worker_prefetch_multiplier=1` in Celery config

---

## 🛠️ ADVANCED CONFIGURATION (OPTIONAL)

### Force Even Distribution Across Workers

Add to `server_production.py`:
```python
celery.conf.update(
    worker_prefetch_multiplier=1,  # Don't prefetch multiple tasks
    task_acks_late=True,           # Acknowledge after completion
    task_reject_on_worker_lost=True,  # Requeue if worker dies
)
```

### Dedicated Segment Queue
```python
# Create separate queue for segment tasks
celery.conf.update(
    task_routes={
        'watermark.process_segment': {'queue': 'segments'},
        'watermark.prepare_video': {'queue': 'celery'},
        'watermark.finalize_video': {'queue': 'celery'},
    }
)
```

Start workers:
```bash
# Worker 1 - only process segments
celery -A server_production.celery worker -Q segments --pool=threads --concurrency=1 -n worker1@%h

# Worker 2 - process everything (prepare, segments, finalize)
celery -A server_production.celery worker -Q celery,segments --pool=threads --concurrency=1 -n worker2@%h
```

---

## ✅ SUCCESS CRITERIA

After implementing the fix, you should see:

1. ✅ Both workers pick up segment tasks within seconds of dispatch
2. ✅ Logs show parallel processing (both workers active simultaneously)
3. ✅ Finalize task triggers automatically when last segment completes
4. ✅ Total processing time ~2x faster than solo pool
5. ✅ No "missed heartbeat" or clock drift warnings
6. ✅ Video completes successfully with all segments merged

---

## 📚 REFERENCES

- Celery Pools: https://docs.celeryq.dev/en/stable/userguide/workers.html#pool
- Concurrency: https://docs.celeryq.dev/en/stable/userguide/concurrency/index.html
- Task Routing: https://docs.celeryq.dev/en/stable/userguide/routing.html

---

**Last Updated:** 2025-10-18
**Status:** CRITICAL FIX - Apply immediately to all cloud workers
