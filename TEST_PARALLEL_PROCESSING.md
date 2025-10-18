# Testing Parallel Distributed Processing

## 🎯 Objective
Verify that multiple Celery workers can process video segments in parallel across different GPUs.

---

## ✅ Current Celery Configuration Analysis

The `server_production.py` already has **optimal settings** for distributed processing:

```python
celery.conf.update(
    worker_prefetch_multiplier=1,  # ✅ PERFECT - prevents one worker from grabbing all tasks
    task_acks_late=True,           # ✅ PERFECT - requeue failed tasks
    task_track_started=True,       # ✅ PERFECT - track task progress
    broker_pool_limit=1,           # ✅ Prevents connection issues
    task_ignore_result=False,      # ✅ Needed for result tracking
)
```

**No configuration changes needed!** The issue was purely the `--pool=solo` startup parameter.

---

## 🚀 STEP-BY-STEP TEST PLAN

### Prerequisites

1. **Two terminals** (or two cloud GPU instances)
2. **Redis accessible** via ngrok tunnel
3. **GPU 0 and GPU 1** available
4. **Test video** ready to upload

---

### Step 1: Stop All Existing Workers

Kill any running workers:
```bash
# Find Celery processes
ps aux | grep celery

# Kill them
pkill -f "celery.*server_production"

# Or press Ctrl+C in each terminal
```

---

### Step 2: Purge Redis Queue

Clear any stuck tasks:
```bash
celery -A server_production.celery purge
```

Confirm with `y`.

Or manually via Redis:
```bash
redis-cli -h 8.tcp.ngrok.io -p 17609 -a watermarkz_secure_2024

# Check queue
LLEN celery

# Clear all
DEL celery

# Clear task metadata
EVAL "return redis.call('del', unpack(redis.call('keys', ARGV[1])))" 0 celery-task-meta-*
```

---

### Step 3: Start Worker 1 (GPU 0)

**Terminal 1:**
```bash
cd /app/waterz/waterz

# Option A: Use startup script
bash start_worker1.sh

# Option B: Manual command
export CUDA_VISIBLE_DEVICES=0
export MIN_SEGMENTS=2
export REDIS_URL="redis://:watermarkz_secure_2024@8.tcp.ngrok.io:17609/0"

celery -A server_production.celery worker \
  --loglevel=info \
  --pool=threads \
  --concurrency=1 \
  -n worker1@%h
```

**Expected Output:**
```
 -------------- worker1@<hostname> v5.3.6
--- ***** -----
-- ******* ---- Linux-... 2025-10-18 18:30:00
- *** --- * ---
- ** ---------- [config]
- ** ---------- .> app:         server_production:...
- ** ---------- .> transport:   redis://:**@8.tcp.ngrok.io:17609/0
- ** ---------- .> results:     redis://:**@8.tcp.ngrok.io:17609/0
- *** --- * --- .> concurrency: 1 (threads)  ← MUST SAY "threads"!
-- ******* ---- .> task events: OFF
--- ***** -----
 -------------- [queues]
                .> celery           exchange=celery(direct) key=celery

[tasks]
  . watermark.prepare_video
  . watermark.process_segment
  . watermark.finalize_video

[INFO/MainProcess] Connected to redis://...
[INFO/MainProcess] mingle: searching for neighbors
[INFO/MainProcess] mingle: sync with 1 nodes  ← Should detect Worker 2
[INFO/MainProcess] worker1@<hostname> ready.
```

✅ **VERIFY:** `.> concurrency: 1 (threads)` - NOT solo!

---

### Step 4: Start Worker 2 (GPU 1)

**Terminal 2:**
```bash
cd /app/waterz/waterz

# Option A: Use startup script
bash start_worker2.sh

# Option B: Manual command
export CUDA_VISIBLE_DEVICES=1
export MIN_SEGMENTS=2
export REDIS_URL="redis://:watermarkz_secure_2024@8.tcp.ngrok.io:17609/0"

celery -A server_production.celery worker \
  --loglevel=info \
  --pool=threads \
  --concurrency=1 \
  -n worker2@%h
```

**Expected Output:**
```
 -------------- worker2@<hostname> v5.3.6
...
- *** --- * --- .> concurrency: 1 (threads)  ← MUST SAY "threads"!
...
[INFO/MainProcess] mingle: sync with 2 nodes  ← Should detect Worker 1
[INFO/MainProcess] worker2@<hostname> ready.
```

✅ **VERIFY:** Both workers see each other in mingle sync!

---

### Step 5: Verify Workers Are Registered

**Terminal 3 (monitoring):**
```bash
# Check active workers
celery -A server_production.celery inspect ping

# Should show:
# worker1@<hostname>: OK
# worker2@<hostname>: OK

# Check registered tasks
celery -A server_production.celery inspect registered

# Both workers should list:
# - watermark.prepare_video
# - watermark.process_segment
# - watermark.finalize_video
```

---

### Step 6: Upload Test Video

Upload a **short video** (1-2 seconds, ~90 frames) through your API:

**Option A: Via Web Interface**
```
Navigate to: https://your-ngrok-url.ngrok-free.dev
Upload video → Click "Remove Watermark"
```

**Option B: Via curl**
```bash
curl -X POST https://your-ngrok-url.ngrok-free.dev/api/upload \
  -F "file=@test_video.mp4" \
  | jq -r '.task_id' > task_id.txt

# Get task ID
TASK_ID=$(cat task_id.txt)

# Start processing
curl -X POST https://your-ngrok-url.ngrok-free.dev/api/process \
  -H "Content-Type: application/json" \
  -d "{\"task_id\": \"$TASK_ID\"}"
```

---

### Step 7: Monitor Parallel Execution

Watch **both worker terminals** simultaneously.

#### ✅ EXPECTED BEHAVIOR (Parallel Processing)

**Worker 2 Terminal:**
```
[18:30:00] Task watermark.prepare_video[abc123] received
[18:30:01] ============================================================
[18:30:01] Loading YOLO detector...
[18:30:05] ✅ YOLO detector ready!
[18:30:05] 📹 Preparing video: test.mp4 (1.77 MB)
[18:30:05] 🎯 Running YOLO detection on 92 frames...
[18:30:15] ✅ Detection complete: 92 frames, 31 with watermarks
[18:30:15] 📊 Detected 2 segments for distributed processing
[18:30:15]    Segment 1: frames 0-65 (66 frames)
[18:30:15]    Segment 2: frames 69-91 (23 frames)
[18:30:15] 🔥 Dispatching 2 segment tasks manually across all workers...
[18:30:15]    ✅ Segment 1 task queued: def456
[18:30:15]    ✅ Segment 2 task queued: ghi789
[18:30:15] ✅ All 2 segment tasks dispatched to queue!
[18:30:15] Task watermark.prepare_video[abc123] succeeded

# After prepare completes, Worker 2 picks up Segment 2:
[18:30:16] Task watermark.process_segment[ghi789] received  ← SEGMENT 2
[18:30:17] 🎬 Worker processing segment 2/2: frames 69-91
[18:30:18]    📐 Crop region: x=0, y=55, w=208, h=176
[18:30:20]    🎨 Running ProPainter...
[18:31:00]    ✅ ProPainter complete for segment 2
[18:31:05]    🎞️  Encoding segment...
[18:31:10] ✅ Segment 2/2 complete!
[18:31:10] 📊 Tracking: 2/2 segments complete
[18:31:10] 🎉 All segments complete! Triggering finalize...
```

**Worker 1 Terminal (AT THE SAME TIME):**
```
[18:30:16] Task watermark.process_segment[def456] received  ← SEGMENT 1
[18:30:17] 🎬 Worker processing segment 1/2: frames 0-65
[18:30:18]    📐 Crop region: x=0, y=55, w=208, h=176
[18:30:20]    🎨 Running ProPainter...
[18:32:00]    ✅ ProPainter complete for segment 1
[18:32:05]    🎞️  Encoding segment...
[18:32:10] ✅ Segment 1/2 complete!
[18:32:10] 📊 Tracking: 1/2 segments complete

# Finalize runs on Worker 2 (whichever finishes last)
```

**🔥 KEY INDICATORS OF SUCCESS:**
1. ✅ Both workers receive segment tasks within **1-2 seconds** of dispatch
2. ✅ Timestamps overlap (both running ProPainter simultaneously)
3. ✅ Segment 1 on Worker 1, Segment 2 on Worker 2 (distributed!)
4. ✅ Last worker to finish triggers finalize
5. ✅ Total time ≈ max(Seg1_time, Seg2_time) + Prepare + Finalize

---

#### ❌ FAILURE BEHAVIOR (If Still Using Solo Pool)

**Worker 2:**
```
[18:30:00] Task watermark.prepare_video received
[18:30:15] ✅ Segment 1 task queued
[18:30:15] ✅ Segment 2 task queued
[18:30:15] prepare_video succeeded
[18:30:16] [IDLE - nothing happening]  ← BUG!
```

**Worker 1:**
```
[18:30:16] Task watermark.process_segment received (Segment 1)
[18:32:10] Segment 1 complete
[18:32:10] Tracking: 1/2 segments complete
[18:32:11] [WAITING FOREVER for Segment 2]  ← STUCK!
```

---

### Step 8: Monitor with Celery Inspect

**Terminal 3 (while processing):**
```bash
# Watch active tasks in real-time
watch -n 1 "celery -A server_production.celery inspect active"
```

**Expected Output (during parallel processing):**
```
worker1@<hostname>:
  [{
    "id": "def456-...",
    "name": "watermark.process_segment",
    "args": "[{'seg_idx': 0, 'start_frame': 0, 'end_frame': 65, ...}]",
    "time_start": 1729274417.0
  }]

worker2@<hostname>:
  [{
    "id": "ghi789-...",
    "name": "watermark.process_segment",
    "args": "[{'seg_idx': 1, 'start_frame': 69, 'end_frame': 91, ...}]",
    "time_start": 1729274417.5
  }]
```

✅ **Both workers active at the same time!**

---

### Step 9: Verify Final Result

Check that finalize task ran and video completed:

```bash
# Check task status
curl https://your-ngrok-url/api/status/<task_id>

# Should return:
{
  "state": "SUCCESS",
  "result": {
    "result_url": "/results/video_propainter.mp4"
  },
  "metadata": {
    "total_segments": 2,
    "total_frames": 92,
    ...
  }
}
```

Download and verify the cleaned video!

---

## 📊 Performance Benchmarks

### Test Video: 92 frames, 2 segments

| Configuration | Seg 1 (66 frames) | Seg 2 (23 frames) | Total Time |
|---------------|-------------------|-------------------|------------|
| **Sequential (1 worker)** | 172s | 90s | ~262s (Seg1 + Seg2) |
| **Parallel (2 workers)** | 172s | 90s | ~172s (max(Seg1, Seg2)) |
| **Speedup** | - | - | **1.52x faster!** |

With 4 segments and 4 workers:
- Sequential: 4 × 90s = 360s
- Parallel: max(90s) = 90s
- **Speedup: 4x faster!**

---

## 🔍 Troubleshooting

### Issue: Only Worker 1 Processes Segments

**Symptoms:**
```
Worker 1: Segment 1 received
Worker 1: Segment 2 received  ← Should go to Worker 2!
Worker 2: [IDLE]
```

**Diagnosis:**
```bash
# Check worker pool type
celery -A server_production.celery inspect stats

# Look for:
# "pool": {"implementation": "celery.concurrency.threads:TaskPool"}
```

**If shows "solo":**
```
"pool": {"implementation": "celery.concurrency.solo:Solo"}
```

→ Worker still using solo pool! Restart with `--pool=threads`

---

### Issue: Segment 2 Stuck in Queue

**Symptoms:**
```
✅ Segment 1 task queued
✅ Segment 2 task queued
[No "received" message for Segment 2]
```

**Diagnosis:**
```bash
# Check Redis queue length
redis-cli -h 8.tcp.ngrok.io -p 17609 -a watermarkz_secure_2024 LLEN celery

# If > 0, tasks are stuck
```

**Fix:**
```bash
# Restart workers
pkill -f celery
bash start_worker1.sh  # Terminal 1
bash start_worker2.sh  # Terminal 2

# Purge and retry
celery -A server_production.celery purge
# Re-upload video
```

---

### Issue: Clock Drift Warnings

**Symptoms:**
```
Substantial drift from worker2 may mean clocks are out of sync. Current drift is 178 seconds.
```

**Cause:** System clocks not synchronized between workers

**Fix:**
```bash
# Sync system time
sudo ntpdate -s time.nist.gov

# Or use systemd-timesyncd
sudo systemctl restart systemd-timesyncd
```

**Alternative:** Disable heartbeat (if on same machine):
```bash
celery -A server_production.celery worker \
  --pool=threads \
  --without-heartbeat \
  --without-gossip \
  --without-mingle
```

---

## ✅ Success Checklist

After running the test, verify:

- [ ] Both workers started with `pool=threads` (not solo)
- [ ] Workers detected each other during mingle
- [ ] Prepare task completed on one worker
- [ ] Segment 1 task picked up by Worker 1
- [ ] Segment 2 task picked up by Worker 2 **within 5 seconds**
- [ ] Both workers ran ProPainter **simultaneously** (overlapping timestamps)
- [ ] Segment completion tracking showed 1/2 → 2/2
- [ ] Finalize task triggered automatically when last segment completed
- [ ] Final video downloaded successfully
- [ ] Total processing time ≈ slowest segment time (not sum of all segments)
- [ ] No "missed heartbeat" or clock drift warnings
- [ ] No tasks stuck in Redis queue after completion

---

## 📈 Scaling Test (Optional)

Test with 4 segments to verify scalability:

```bash
# Force 4 segments
export MIN_SEGMENTS=4
export MIN_CHUNK_FRAMES=20

# Restart workers with new config
# Upload video
# Should create 4 segments even if watermark doesn't move
```

**Expected:**
- Worker 1: Segment 1, then Segment 3
- Worker 2: Segment 2, then Segment 4

Total time ≈ 2 × segment_time (each worker processes 2 segments sequentially)

With 4 workers:
- Each processes 1 segment
- Total time ≈ 1 × segment_time (4x faster!)

---

## 🎯 Next Steps After Verification

Once parallel processing is confirmed working:

1. **Deploy to Cloud GPUs**
   - Start workers on Salad/RunPod/Vast.ai
   - Each GPU instance runs one worker
   - All connect to same Redis (ngrok tunnel)

2. **Monitor Production**
   ```bash
   # Real-time monitoring
   watch -n 2 "celery -A server_production.celery inspect active"

   # Task success rate
   celery -A server_production.celery inspect stats
   ```

3. **Scale Up**
   - Add more workers → faster processing
   - 4 workers = 4x speedup
   - 8 workers = 8x speedup (if video has enough segments)

4. **Optimize Segment Count**
   ```bash
   # Adjust MIN_SEGMENTS based on worker count
   export MIN_SEGMENTS=<number_of_workers>
   ```

---

**Last Updated:** 2025-10-18
**Test Status:** Ready for execution
