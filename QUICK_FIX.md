# 🚨 QUICK FIX - Segments Not Processing in Parallel

## THE PROBLEM
Workers using `--pool=solo` **cannot pick up new tasks** while processing.

**Symptom:** Segment 2 never gets processed, job hangs forever.

---

## THE SOLUTION (2 minutes)

### 1. Kill Both Workers
```bash
# Press Ctrl+C in both worker terminals
# Or:
pkill -f "celery.*server_production"
```

### 2. Purge Stuck Tasks
```bash
celery -A server_production.celery purge
# Type: y
```

### 3. Restart with Threads Pool

**Terminal 1 - Worker 1:**
```bash
export CUDA_VISIBLE_DEVICES=0
export REDIS_URL="redis://:watermarkz_secure_2024@8.tcp.ngrok.io:17609/0"

celery -A server_production.celery worker \
  --loglevel=info \
  --pool=threads \
  --concurrency=1 \
  -n worker1@%h
```

**Terminal 2 - Worker 2:**
```bash
export CUDA_VISIBLE_DEVICES=1
export REDIS_URL="redis://:watermarkz_secure_2024@8.tcp.ngrok.io:17609/0"

celery -A server_production.celery worker \
  --loglevel=info \
  --pool=threads \
  --concurrency=1 \
  -n worker2@%h
```

### 4. Verify It Says "threads"
```
- *** --- * --- .> concurrency: 1 (threads)  ← MUST SEE THIS!
```

❌ If you see `(solo)` → Wrong pool, restart!

---

## VERIFY IT WORKS

Upload a test video and watch logs:

**✅ GOOD - Parallel Processing:**
```
Worker 1: [18:30:16] process_segment[seg1] received
Worker 2: [18:30:16] process_segment[seg2] received  ← Both at same time!
```

**❌ BAD - Sequential/Stuck:**
```
Worker 1: [18:30:16] process_segment[seg1] received
Worker 1: [18:32:10] Segment 1 complete
Worker 1: [18:32:11] Tracking: 1/2 segments complete
Worker 2: [IDLE - nothing]  ← Segment 2 never picked up!
```

---

## DONE!

That's it. Now segments process in parallel across both workers.

**For detailed explanation, read:** `DISTRIBUTED_PROCESSING_FIX.md`
**For testing guide, read:** `TEST_PARALLEL_PROCESSING.md`
