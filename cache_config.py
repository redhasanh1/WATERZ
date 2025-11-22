"""
Enables per-worker PyTorch inductor cache isolation for multi-worker safety.

This module is imported early in watermark.py to set up isolated cache directories
for each worker process (supports both Celery and ProcessPoolExecutor).

Issue: PyTorch's inductor cache can cause file locking and corruption
when multiple processes try to write to the same cache directory.

Solution: Each worker gets its own cache directory based on process ID.
- Worker 0 → .torch_cache_0
- Worker 1 → .torch_cache_1
- Worker 2 → .torch_cache_2
- Worker 3 → .torch_cache_3

Supports:
- Celery multiprocessing pool (uses _identity tuple)
- ProcessPoolExecutor (uses process name like 'SpawnProcess-1')
- ThreadPoolExecutor (shared cache OK due to GIL)

This enables TRUE parallel processing with torch.compile!
"""
import os
import multiprocessing
import re

def get_worker_id():
    """
    Detect worker ID from multiprocessing context.

    For ProcessPoolExecutor, workers REUSE the same process (SpawnProcess-1) for sequential tasks,
    so we can't rely on process name/ID. Instead, we use a global counter that increments per cache setup.

    Returns:
        int: Worker number (0-indexed), or 0 for main process
    """
    process = multiprocessing.current_process()

    # Method 1: Check for environment variable (set by caller if using parallel workers)
    env_worker_id = os.getenv('WORKER_ID', None)
    if env_worker_id is not None:
        return int(env_worker_id)

    # Method 2: Celery-style _identity tuple (e.g., (1,) (2,) (3,))
    # This works for Celery multiprocessing pool where each worker is a separate process
    worker_id = getattr(process, '_identity', None)
    if worker_id and len(worker_id) > 0:
        worker_num = worker_id[0] - 1  # Convert 1-indexed to 0-indexed
        return worker_num

    # Method 3: ProcessPoolExecutor - Use process ID hash for cache isolation
    # ProcessPoolExecutor reuses processes, so we use PID to ensure different cache per process
    # This ensures that even if the same process is reused, it gets a consistent cache
    process_name = process.name
    if process_name and ('SpawnProcess' in process_name or 'ForkProcess' in process_name):
        # Use process PID modulo 8 to get worker ID (supports up to 8 workers)
        # This ensures the same process always uses the same cache
        worker_num = os.getpid() % 8
        return worker_num

    # Default: Main process or single-worker
    return 0

# Get worker ID using detection logic
worker_num = get_worker_id()
cache_dir = os.path.join(os.path.dirname(__file__), 'temp', f'.torch_cache_{worker_num}')
triton_cache_dir = os.path.join(os.path.dirname(__file__), 'temp', f'.triton_cache_{worker_num}')

# Create cache directories
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(triton_cache_dir, exist_ok=True)

# Set per-worker cache directories (only if not already set by batch file)
if 'TORCHINDUCTOR_CACHE_DIR' not in os.environ:
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = cache_dir
if 'TRITON_CACHE_DIR' not in os.environ:
    os.environ['TRITON_CACHE_DIR'] = triton_cache_dir

# Enable FX graph cache and autotune cache (set in START_CELERY_TRT.bat)
# os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] is already set by batch file
# os.environ['TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE'] is already set by batch file

process_name = multiprocessing.current_process().name
print(f"[cache_config] Worker {worker_num} ({process_name}): {cache_dir}")
