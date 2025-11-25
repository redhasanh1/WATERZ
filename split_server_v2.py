"""
Smart script to split server_production.py
Keeps: Flask routes, imports, config
Removes: Celery tasks, GPU functions
"""
import re

def parse_and_split():
    with open('server_production.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    api_lines = []
    tasks_lines = []

    # State tracking
    in_task_function = False
    in_gpu_function = False
    in_route_function = False
    task_function_indent = 0
    gpu_function_names = ['get_detector', 'get_propainter_pipeline', '_ensure_cuda_torch',
                          'background_encoder_worker', 'encode_segment_background',
                          'trigger_finalization', 'start_background_encoder',
                          'start_redis_download_poller', 'cleanup_old_files',
                          'schedule_cleanup', 'on_worker_ready']

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for GPU function definitions
        if any(f'def {name}(' in line for name in gpu_function_names):
            in_gpu_function = True
            gpu_function_indent = len(line) - len(line.lstrip())
            tasks_lines.append(line)
            i += 1
            continue

        # Check for @celery.task decorator
        if '@celery.task' in line or in_task_function:
            if '@celery.task' in line:
                in_task_function = True
                task_function_indent = len(line) - len(line.lstrip())
            tasks_lines.append(line)

            # End of task function when we hit a line at same/lower indent that's not blank/comment
            if in_task_function and i > 0:
                current_indent = len(line) - len(line.lstrip())
                if line.strip() and not line.strip().startswith('#') and current_indent <= task_function_indent and 'def ' in line and i > task_function_indent:
                    in_task_function = False
            i += 1
            continue

        # Check for @worker_ready decorator
        if '@worker_ready' in line:
            in_gpu_function = True
            gpu_function_indent = len(line) - len(line.lstrip())
            tasks_lines.append(line)
            i += 1
            continue

        # Inside GPU function - copy to tasks only
        if in_gpu_function:
            tasks_lines.append(line)
            # End GPU function at next function definition or lower indent
            current_indent = len(line) - len(line.lstrip())
            if line.strip() and not line.strip().startswith('#'):
                if 'def ' in line and current_indent <= gpu_function_indent and i > 10:
                    in_gpu_function = False
            i += 1
            continue

        # Everything else goes to API file
        api_lines.append(line)
        i += 1

    # Write server_api.py
    with open('server_api.py', 'w', encoding='utf-8') as f:
        f.write('"""\\nFlask API Server (Railway) - No GPU Code\\n"""\\n\\n')
        f.writelines(api_lines)

    # Write celery_tasks.py header
    with open('celery_tasks.py', 'w', encoding='utf-8') as f:
        f.write('"""\\nCelery Worker Tasks - GPU Processing\\n"""\\n\\n')
        # Copy imports and config (first 630 lines)
        f.writelines(lines[:630])
        f.write('\\n# ===== CELERY TASKS AND GPU FUNCTIONS =====\\n\\n')
        f.writelines(tasks_lines)

    print(f"[OK] server_api.py: {len(api_lines)} lines")
    print(f"[OK] celery_tasks.py: {630 + len(tasks_lines)} lines")

if __name__ == '__main__':
    import os
    os.chdir(r'D:\\watermarkz')
    parse_and_split()
    print("\\n[SUCCESS] Split complete!")
