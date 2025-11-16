#!/usr/bin/env python3
"""Add debug logging to SAM2 endpoint"""

with open('web/app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the line numbers we need to modify
for i, line in enumerate(lines):
    # Add log after REDIS_URL check
    if 'redis_client = redis.from_url(REDIS_URL, decode_responses=False)' in line:
        # Insert log before this line
        indent = '        '
        lines.insert(i, f'{indent}print(f"[SAM2] Using Redis: {{REDIS_URL[:50]}}...")\n')
        print(f"[OK] Added Redis URL logging at line {i+1}")
        break

# Find publish line
for i, line in enumerate(lines):
    if "redis_client.publish('sam2:selection:request'," in line:
        # Insert logs before this line
        indent = '        '
        lines.insert(i, f'{indent}print(f"[SAM2] Publishing request {{request_id}} to channel: sam2:selection:request")\n')
        lines.insert(i+1, f'{indent}print(f"[SAM2] Request data: points={{len(points)}}, video={{video_width}}x{{video_height}}")\n')
        print(f"[OK] Added publish logging at line {i+1}")
        break

# Find response wait line
for i, line in enumerate(lines):
    if 'while time.time() - start_time < timeout:' in line:
        # Insert log before this line
        indent = '        '
        lines.insert(i, f'{indent}print(f"[SAM2] Waiting for response on: {{response_channel}}")\n')
        print(f"[OK] Added response wait logging at line {i+1}")
        break

with open('web/app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("[OK] Added all debug logging to web/app.py")
