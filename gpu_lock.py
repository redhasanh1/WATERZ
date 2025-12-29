"""
Simple GPU lock - user-triggered release.
Lock is acquired by task, released when user acknowledges download.
"""
import redis
import os

# Load Redis URL (same logic as workers)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
redis_path = os.path.join(BASE_DIR, 'redis_url.txt')
if os.path.exists(redis_path):
    with open(redis_path, 'r') as f:
        REDIS_URL = f.read().strip()

GPU_LOCK_KEY = "gpu_lock:holder"
GPU_LOCK_TIMEOUT = 600  # 10 min auto-expire (safety)


def acquire_lock(job_id, max_wait=300):
    """
    Try to acquire GPU lock. Blocks until lock is free or timeout.
    Returns True if acquired, raises TimeoutError if not.
    """
    import time
    r = redis.from_url(REDIS_URL)
    start = time.time()

    while time.time() - start < max_wait:
        # Try to set lock (only if not exists)
        if r.set(GPU_LOCK_KEY, job_id, nx=True, ex=GPU_LOCK_TIMEOUT):
            print(f"[GPU-LOCK] Acquired by {job_id}")
            return True

        # Lock held - check who has it
        holder = r.get(GPU_LOCK_KEY)
        if holder:
            holder = holder.decode() if isinstance(holder, bytes) else holder
            print(f"[GPU-LOCK] Waiting... held by {holder}")

        time.sleep(2)

    raise TimeoutError(f"GPU lock timeout after {max_wait}s")


def release_lock(job_id):
    """
    Release GPU lock if we own it.
    Called by frontend when user receives download URL.
    """
    r = redis.from_url(REDIS_URL)
    holder = r.get(GPU_LOCK_KEY)

    if holder:
        holder = holder.decode() if isinstance(holder, bytes) else holder
        if holder == job_id:
            r.delete(GPU_LOCK_KEY)
            print(f"[GPU-LOCK] Released by {job_id}")
            return True
        else:
            print(f"[GPU-LOCK] Cannot release - held by {holder}, not {job_id}")
            return False
    else:
        print(f"[GPU-LOCK] No lock to release")
        return True


def force_release():
    """Force release lock (admin use only)."""
    r = redis.from_url(REDIS_URL)
    r.delete(GPU_LOCK_KEY)
    print(f"[GPU-LOCK] Force released")
    return True


def get_lock_status():
    """Get current lock holder."""
    r = redis.from_url(REDIS_URL)
    holder = r.get(GPU_LOCK_KEY)
    if holder:
        holder = holder.decode() if isinstance(holder, bytes) else holder
        ttl = r.ttl(GPU_LOCK_KEY)
        return {'locked': True, 'holder': holder, 'ttl': ttl}
    return {'locked': False}
