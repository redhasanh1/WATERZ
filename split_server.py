"""
Script to split server_production.py into server_api.py and celery_tasks.py
"""

# Line ranges to EXCLUDE from server_api.py (GPU code):
# - Lines 81-141: _ensure_cuda_torch() function
# - Lines 1834-1982: background_encoder_worker() and related
# - Lines 2454-2638: Redis download poller and cleanup threads
# - Lines 2525-2601: Worker initialization (@worker_ready)
# - Lines 2691-2724: get_detector()
# - Lines 2725-2777: get_propainter_pipeline()
# - Lines 2778-end of tasks: All @celery.task definitions

GPU_EXCLUDE_RANGES = [
    (81, 141),      # _ensure_cuda_torch
    (1834, 2638),   # background threads and worker init
    (2691, 7000),   # get_detector, get_propainter, all tasks (keep Flask routes only)
]

def should_exclude_from_api(line_num):
    """Check if line should be excluded from server_api.py"""
    for start, end in GPU_EXCLUDE_RANGES:
        if start <= line_num <= end:
            return True
    return False

def create_server_api():
    """Create server_api.py - Flask only, no GPU"""
    print("Creating server_api.py...")
    with open('server_production.py', 'r', encoding='utf-8') as f_in:
        with open('server_api.py', 'w', encoding='utf-8') as f_out:
            f_out.write('"""\nFlask API Server for Railway (no GPU code)\nGenerated from server_production.py\n"""\n\n')

            line_num = 0
            for line in f_in:
                line_num += 1
                if not should_exclude_from_api(line_num):
                    f_out.write(line)
                elif line_num == 81:
                    # Replace GPU init with stub
                    f_out.write('# GPU initialization skipped (API-only mode)\n')
                    f_out.write('GPU_AVAILABLE = False\n\n')
                elif line_num == 1834:
                    f_out.write('# Background encoder and worker threads skipped (API-only mode)\n\n')
                elif line_num == 2691:
                    f_out.write('# GPU functions and Celery tasks excluded (see celery_tasks.py)\n\n')

    print(f"[OK] Created server_api.py")

def create_celery_tasks():
    """Create celery_tasks.py - Workers only, GPU code"""
    print("Creating celery_tasks.py...")
    with open('server_production.py', 'r', encoding='utf-8') as f_in:
        with open('celery_tasks.py', 'w', encoding='utf-8') as f_out:
            f_out.write('"""\nCelery Worker Tasks (GPU processing)\nGenerated from server_production.py\n"""\n\n')

            # Copy imports and config (lines 1-630)
            f_in.seek(0)
            for i, line in enumerate(f_in, 1):
                if i <= 630:
                    f_out.write(line)
                elif i == 631:
                    break

            # Skip Flask routes (lines 631-2690)
            f_in.seek(0)
            for i, line in enumerate(f_in, 1):
                if i < 2691:
                    continue
                f_out.write(line)

    print(f"[OK] Created celery_tasks.py")

if __name__ == '__main__':
    import os
    os.chdir(r'D:\watermarkz')

    create_server_api()
    create_celery_tasks()

    print("\n[SUCCESS] Split complete!")
    print("  - server_api.py: Flask routes, no GPU")
    print("  - celery_tasks.py: Celery tasks with GPU")
