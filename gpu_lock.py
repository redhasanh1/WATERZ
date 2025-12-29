"""
Redis-based GPU lock for coordinating Celery workers.
Ensures only one GPU task runs at a time across containers.
Uses pub/sub for instant wake-up (no polling).
"""
import redis
import time
import os

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
GPU_LOCK_KEY = "gpu_lock:exclusive"
GPU_LOCK_CHANNEL = "gpu_lock:released"
GPU_LOCK_TIMEOUT = 600  # 10 min max (auto-release if crash)


class GPULock:
    """Context manager - blocks until GPU is free using pub/sub (no polling)."""

    def __init__(self, worker_id, max_wait=300):
        self.worker_id = worker_id
        self.max_wait = max_wait
        self.r = redis.from_url(REDIS_URL)

    def __enter__(self):
        # Try to acquire immediately
        if self.r.set(GPU_LOCK_KEY, self.worker_id, nx=True, ex=GPU_LOCK_TIMEOUT):
            print(f"[GPU-LOCK] Acquired by {self.worker_id}")
            return self

        # Lock held - subscribe and wait for release signal
        print(f"[GPU-LOCK] Waiting for release signal...")
        pubsub = self.r.pubsub()
        pubsub.subscribe(GPU_LOCK_CHANNEL)

        start = time.time()
        while time.time() - start < self.max_wait:
            # Wait for message with 5s timeout (fallback safety)
            msg = pubsub.get_message(timeout=5)

            # Try to acquire after any wake-up
            if self.r.set(GPU_LOCK_KEY, self.worker_id, nx=True, ex=GPU_LOCK_TIMEOUT):
                pubsub.close()
                print(f"[GPU-LOCK] Acquired by {self.worker_id}")
                return self

        pubsub.close()
        raise TimeoutError(f"GPU lock timeout after {self.max_wait}s")

    def __exit__(self, *args):
        if self.r.get(GPU_LOCK_KEY) == self.worker_id.encode():
            self.r.delete(GPU_LOCK_KEY)
            # Notify waiting tasks instantly
            self.r.publish(GPU_LOCK_CHANNEL, "released")
            print(f"[GPU-LOCK] Released by {self.worker_id}")
