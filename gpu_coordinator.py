"""
GPU Coordinator - Redis-based distributed lock for GPU exclusivity.

Ensures only ONE Celery worker (SAM2 or ProPainter) uses the GPU at a time.
Workers wait (pause) while another worker has the lock - tasks stay in queue.

Usage:
    from gpu_coordinator import GPUCoordinator

    coord = GPUCoordinator(worker_type='sam2')  # or 'propainter'

    with coord.gpu_exclusive(task_id):
        # GPU-intensive work here
        ...
"""

import redis
import time
import threading
import os
import json
from contextlib import contextmanager

# Configuration from environment
REDIS_URL = os.getenv('REDIS_URL')
LOCK_TTL = int(os.getenv('GPU_LOCK_TTL', '120'))  # 2 minutes - auto-expire if worker dies
HEARTBEAT_INTERVAL = int(os.getenv('GPU_HEARTBEAT_INTERVAL', '30'))  # Extend TTL every 30s
POLL_INTERVAL = 2  # Check lock every 2 seconds

# Redis keys
LOCK_KEY = 'gpu:lock'
LAST_TYPE_KEY = 'gpu:lock:last_type'


class GPUCoordinator:
    """
    Coordinates GPU access between SAM2 and ProPainter Celery workers.

    Features:
    - Redis SETNX lock with TTL (auto-expires on crash)
    - Heartbeat thread extends TTL for long tasks
    - Fair scheduling (alternates between worker types when both waiting)
    """

    def __init__(self, worker_type: str, worker_id: str = None):
        """
        Initialize GPU coordinator.

        Args:
            worker_type: 'sam2' or 'propainter'
            worker_id: Unique identifier (defaults to type + PID)
        """
        if not REDIS_URL:
            raise RuntimeError("REDIS_URL environment variable not set")

        self.redis = redis.from_url(REDIS_URL, decode_responses=True)
        self.worker_type = worker_type
        self.worker_id = worker_id or f"{worker_type}-{os.getpid()}"
        self._heartbeat_thread = None
        self._stop_heartbeat = threading.Event()

    def _lock_value(self, task_id: str) -> str:
        """Create JSON lock value with metadata."""
        return json.dumps({
            'worker_type': self.worker_type,
            'worker_id': self.worker_id,
            'task_id': task_id,
            'acquired_at': time.time(),
            'last_heartbeat': time.time()
        })

    def _other_type_waiting(self) -> bool:
        """Check if the other worker type has tasks waiting."""
        other = 'propainter' if self.worker_type == 'sam2' else 'sam2'
        waiting = self.redis.get(f'gpu:lock:waiting:{other}')
        return waiting is not None and int(waiting) > 0

    def acquire_lock(self, task_id: str, timeout: int = 3600) -> bool:
        """
        Acquire GPU lock with fair scheduling.

        Blocks until lock is acquired or timeout.
        Uses fair scheduling to alternate between worker types.

        Args:
            task_id: Celery task ID
            timeout: Max seconds to wait (default 1 hour)

        Returns:
            True if lock acquired, False if timeout
        """
        start_time = time.time()
        log_interval = 10  # Log waiting status every 10 seconds
        last_log_time = 0
        yield_start_time = None  # Track how long we've been yielding
        MAX_YIELD_TIME = 10  # Stop yielding after 10 seconds if lock is free

        # Register in waiting queue
        self.redis.incr(f'gpu:lock:waiting:{self.worker_type}')

        try:
            while time.time() - start_time < timeout:
                # Check if lock is currently held
                lock_is_free = self.redis.get(LOCK_KEY) is None

                # Check fair scheduling - yield if we just ran and other type is waiting
                last_holder = self.redis.get(LAST_TYPE_KEY)
                other_waiting = self._other_type_waiting()
                should_yield = (
                    last_holder == self.worker_type and
                    other_waiting and
                    lock_is_free  # Only yield if lock is actually free
                )

                # Track yield time - if we've been yielding too long, stop
                if should_yield:
                    if yield_start_time is None:
                        yield_start_time = time.time()
                    elif time.time() - yield_start_time > MAX_YIELD_TIME:
                        # Other type isn't taking the lock, stop yielding
                        print(f"[GPU-LOCK] Stopped yielding after {MAX_YIELD_TIME}s - other type not responding")
                        should_yield = False
                        yield_start_time = None
                else:
                    yield_start_time = None

                if not should_yield:
                    # Try to acquire lock with SETNX
                    lock_value = self._lock_value(task_id)
                    if self.redis.set(LOCK_KEY, lock_value, nx=True, ex=LOCK_TTL):
                        # Lock acquired!
                        self.redis.set(LAST_TYPE_KEY, self.worker_type)
                        print(f"[GPU-LOCK] Acquired by {self.worker_id} for task {task_id}")
                        self._start_heartbeat(task_id)
                        return True

                # Lock held by someone else - log status periodically
                current_time = time.time()
                if current_time - last_log_time >= log_interval:
                    current = self.redis.get(LOCK_KEY)
                    if current:
                        try:
                            holder = json.loads(current)
                            other_type = holder.get('worker_type', 'unknown')
                            other_task = holder.get('task_id', 'unknown')[:8]
                            wait_time = int(current_time - start_time)
                            print(f"[GPU-LOCK] Waiting {wait_time}s... Lock held by {other_type} (task: {other_task}...)")
                        except json.JSONDecodeError:
                            pass
                    elif should_yield:
                        print(f"[GPU-LOCK] Yielding to other worker type (fair scheduling)")
                    last_log_time = current_time

                # Wait before retry
                time.sleep(POLL_INTERVAL)

            print(f"[GPU-LOCK] TIMEOUT after {timeout}s waiting for lock")
            return False

        finally:
            # Remove from waiting queue
            self.redis.decr(f'gpu:lock:waiting:{self.worker_type}')

    def release_lock(self, task_id: str):
        """
        Release GPU lock after task completion.

        Verifies ownership before releasing to prevent accidental
        release of another worker's lock.
        """
        self._stop_heartbeat.set()

        # Verify we own the lock before releasing
        current = self.redis.get(LOCK_KEY)
        if current:
            try:
                holder = json.loads(current)
                if holder.get('worker_id') == self.worker_id:
                    self.redis.delete(LOCK_KEY)
                    print(f"[GPU-LOCK] Released by {self.worker_id} (task: {task_id})")
                else:
                    print(f"[GPU-LOCK] WARNING: Lock owned by {holder.get('worker_id')}, not releasing")
            except json.JSONDecodeError:
                # Corrupted lock data - safe to delete
                self.redis.delete(LOCK_KEY)
                print(f"[GPU-LOCK] Released (corrupted lock data)")

        # Clean up heartbeat thread
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
            self._heartbeat_thread = None

    def _start_heartbeat(self, task_id: str):
        """Start background thread to extend lock TTL for long tasks."""
        self._stop_heartbeat.clear()

        def heartbeat_loop():
            while not self._stop_heartbeat.wait(HEARTBEAT_INTERVAL):
                try:
                    # Extend lock TTL (only if we still own it)
                    current = self.redis.get(LOCK_KEY)
                    if current:
                        holder = json.loads(current)
                        if holder.get('worker_id') == self.worker_id:
                            lock_value = self._lock_value(task_id)
                            self.redis.set(LOCK_KEY, lock_value, xx=True, ex=LOCK_TTL)
                except Exception as e:
                    print(f"[GPU-LOCK] Heartbeat error: {e}")

        self._heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    @contextmanager
    def gpu_exclusive(self, task_id: str, timeout: int = 3600):
        """
        Context manager for exclusive GPU access.

        Usage:
            with coord.gpu_exclusive(task_id):
                # GPU work here
                ...

        Args:
            task_id: Celery task ID
            timeout: Max seconds to wait for lock

        Raises:
            RuntimeError: If lock cannot be acquired within timeout
        """
        acquired = self.acquire_lock(task_id, timeout=timeout)
        if not acquired:
            raise RuntimeError(f"[GPU-LOCK] Failed to acquire GPU lock for task {task_id} within {timeout}s")
        try:
            yield
        finally:
            self.release_lock(task_id)

    def force_release(self):
        """
        Force release any lock (for admin/debugging).

        WARNING: Only use this for stuck locks after verifying
        no worker is actually using the GPU.
        """
        self.redis.delete(LOCK_KEY)
        print(f"[GPU-LOCK] FORCE RELEASED by {self.worker_id}")

    def get_lock_status(self) -> dict:
        """Get current lock status for monitoring."""
        current = self.redis.get(LOCK_KEY)
        waiting_sam2 = self.redis.get('gpu:lock:waiting:sam2') or '0'
        waiting_propainter = self.redis.get('gpu:lock:waiting:propainter') or '0'

        status = {
            'locked': current is not None,
            'waiting_sam2': int(waiting_sam2),
            'waiting_propainter': int(waiting_propainter),
        }

        if current:
            try:
                holder = json.loads(current)
                status['holder'] = holder
                status['age_seconds'] = time.time() - holder.get('acquired_at', 0)
            except json.JSONDecodeError:
                status['holder'] = 'corrupted'

        return status


# Singleton instances for each worker type
_coordinators = {}

def get_coordinator(worker_type: str) -> GPUCoordinator:
    """Get or create a GPU coordinator for the given worker type."""
    if worker_type not in _coordinators:
        _coordinators[worker_type] = GPUCoordinator(worker_type)
    return _coordinators[worker_type]
