"""
Enables per-worker PyTorch inductor cache isolation for multi-worker safety.

This module is imported early in watermark.py to set up isolated cache directories
for each Celery worker process.

Issue: PyTorch's inductor cache can cause file locking and corruption
when multiple processes try to write to the same cache directory.

Solution: Each worker gets its own cache directory based on process ID.
- Worker 0 → .torch_cache_0
- Worker 1 → .torch_cache_1
- Worker 2 → .torch_cache_2
- Worker 3 → .torch_cache_3

This enables TRUE parallel processing with torch.compile!
"""
import os
import multiprocessing

# Get worker ID from multiprocessing (Celery uses multiprocessing pool)
worker_id = multiprocessing.current_process()._identity
if worker_id and len(worker_id) > 0:
    # Worker process: use worker-specific cache directory
    worker_num = worker_id[0] - 1  # Convert 1-indexed to 0-indexed
    cache_dir = os.path.join(os.path.dirname(__file__), 'temp', f'.torch_cache_{worker_num}')
    triton_cache_dir = os.path.join(os.path.dirname(__file__), 'temp', f'.triton_cache_{worker_num}')
else:
    # Main process or single-worker: use default cache
    cache_dir = os.path.join(os.path.dirname(__file__), 'temp', '.torch_cache_0')
    triton_cache_dir = os.path.join(os.path.dirname(__file__), 'temp', '.triton_cache_0')

# Create cache directories
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(triton_cache_dir, exist_ok=True)

# Set per-worker cache directories
os.environ['TORCHINDUCTOR_CACHE_DIR'] = cache_dir
os.environ['TRITON_CACHE_DIR'] = triton_cache_dir

# Enable FX graph cache and autotune cache (set in START_CELERY_TRT.bat)
# os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] is already set by batch file
# os.environ['TORCHINDUCTOR_AUTOTUNE_LOCAL_CACHE'] is already set by batch file

print(f"[cache_config] Per-worker cache isolation enabled: {cache_dir}")
