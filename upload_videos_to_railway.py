#!/usr/bin/env python3
"""
Upload videos to Railway volume via admin endpoint
"""
import os
import sys
import requests

# Configuration
SITE_URL = "https://markremoverai.com"
ADMIN_SECRET = "c9737e23556159506143180a6ea5137225aedf633079833d8c78a2a33d3fa59b"  # Railway admin secret

# Videos to upload
VIDEOS = [
    {
        'path': 'videostotrain/s2.mp4',
        'type': 'static',
        'description': 'Before watermark removal - end preview'
    },
    {
        'path': 'videostotrain/s2removed.mp4',
        'type': 'static',
        'description': 'After watermark removal - end preview'
    },
    {
        'path': 'videostotrain/cool.mp4',
        'type': 'static',
        'description': 'AI preview - middle section'
    }
]


def upload_video(video_path, video_type, description):
    """Upload a single video to Railway volume"""
    if not os.path.exists(video_path):
        print(f"[X] File not found: {video_path}")
        return False

    filename = os.path.basename(video_path)
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)

    print(f"\n[>>] Uploading: {filename} ({file_size_mb:.1f} MB)")
    print(f"   Type: {video_type}")
    print(f"   Description: {description}")

    try:
        with open(video_path, 'rb') as f:
            files = {'video': (filename, f, 'video/mp4')}
            data = {'type': video_type}
            headers = {'X-Admin-Secret': ADMIN_SECRET}

            response = requests.post(
                f"{SITE_URL}/admin/upload-video",
                files=files,
                data=data,
                headers=headers,
                timeout=300  # 5 min timeout for large files
            )

        if response.status_code == 200:
            result = response.json()
            print(f"[OK] Success!")
            print(f"   URL: {SITE_URL}{result['url']}")
            return True
        else:
            print(f"[FAIL] Upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def main():
    print("=" * 60)
    print("Upload Videos to Railway Volume")
    print("=" * 60)

    # Check if running against production or local
    if len(sys.argv) > 1:
        global SITE_URL
        SITE_URL = sys.argv[1]
        print(f"\n[>>] Target: {SITE_URL}")
    else:
        print(f"\n[>>] Target: {SITE_URL} (production)")
        print("   Tip: Use 'python upload_videos_to_railway.py http://localhost:5000' for local testing")

    # Get admin secret from environment if available
    if os.getenv('ADMIN_SECRET'):
        global ADMIN_SECRET
        ADMIN_SECRET = os.getenv('ADMIN_SECRET')
        print(f"   Using ADMIN_SECRET from environment")
    else:
        print(f"   Using default dev secret")

    # Upload all videos
    success_count = 0
    for video in VIDEOS:
        if upload_video(video['path'], video['type'], video['description']):
            success_count += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"[OK] Uploaded {success_count}/{len(VIDEOS)} videos successfully")
    print("=" * 60)

    if success_count < len(VIDEOS):
        sys.exit(1)


if __name__ == '__main__':
    main()
