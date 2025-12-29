"""
Redis-based GPU lock for coordinating Celery workers.
Ensures only one GPU task runs at a time across containers.
"""
import redis
import time
import os

REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
GPU_LOCK_KEY = "gpu_lock:exclusive"
GPU_LOCK_TIMEOUT = 600  # 10 min max (auto-release if crash)


class GPULock:
    """Context manager - blocks until GPU is free."""

    def __init__(self, worker_id, max_wait=300):
        self.worker_id = worker_id
        self.max_wait = max_wait
        self.r = redis.from_url(REDIS_URL)

    def __enter__(self):
        start = time.time()
        while time.time() - start < self.max_wait:
            if self.r.set(GPU_LOCK_KEY, self.worker_id, nx=True, ex=GPU_LOCK_TIMEOUT):
                print(f"[GPU-LOCK] Acquired by {self.worker_id}")
                return self
            holder = self.r.get(GPU_LOCK_KEY)
            print(f"[GPU-LOCK] Waiting... held by {holder}")
            time.sleep(2)
        raise TimeoutError(f"GPU lock timeout after {self.max_wait}s")

    def __exit__(self, *args):
        if self.r.get(GPU_LOCK_KEY) == self.worker_id.encode():
            self.r.delete(GPU_LOCK_KEY)
            print(f"[GPU-LOCK] Released by {self.worker_id}")
