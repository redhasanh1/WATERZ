import sys
import os

# Ensure repo-root python_packages and repo root are importable regardless of CWD
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_PY_PKGS = os.path.join(_REPO_ROOT, 'python_packages')
for _p in (_PY_PKGS, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from playwright.sync_api import sync_playwright  # type: ignore
    _HAS_PLAYWRIGHT = True
except Exception:
    sync_playwright = None  # type: ignore
    _HAS_PLAYWRIGHT = False
import time
import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from url_utils import sanitize_video_url, is_potentially_watermarked_url
import re
import html
from urllib.parse import urljoin

# Import from parent directory
sys.path.insert(0, '..')
from yolo_detector import YOLOWatermarkDetector
from lama_inpaint_local import LamaInpainter

def download_video(url, output_filename, cookies_file="cookies.json"):
    """Download video using Playwright with saved cookies.

    Falls back to HTTP+cookies parsing if Playwright is unavailable or fails.
    """

    def _http_fallback(page_url: str) -> str:
        print("\n🛟 Falling back to HTTP session scraping...")
        import requests

        if not os.path.exists(cookies_file):
            print("⚠️  No cookies found - run save_cookies.py first!")
            return None

        sess = requests.Session()
        sess.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })

        try:
            with open(cookies_file, 'r') as f:
                cookies = json.load(f)
            for c in cookies:
                domain = c.get('domain') or 'sora.chatgpt.com'
                if 'sora.chatgpt.com' in domain:
                    sess.cookies.set(
                        name=c.get('name'),
                        value=c.get('value'),
                        domain=domain,
                        path=c.get('path', '/'),
                    )
        except Exception as e:
            print(f"❌ Failed to load cookies: {e}")
            return None

        try:
            resp = sess.get(page_url, timeout=30)
            if resp.status_code >= 400:
                print(f"❌ HTTP error fetching page: {resp.status_code}")
                return None
            content = resp.text
        except Exception as e:
            print(f"❌ HTTP error: {e}")
            return None

        # Try to extract .mp4 URLs from HTML
        mp4_urls = re.findall(r'https?://[^\s"\'<>]+\.mp4[^\s"\'<>]*', content)
        video_src = None
        if mp4_urls:
            video_src = html.unescape(mp4_urls[0])
        else:
            # Try to find <video src="...">
            m = re.search(r'<video[^>]+src=["\']([^"\']+)["\']', content, re.IGNORECASE)
            if m:
                video_src = html.unescape(m.group(1))

        if not video_src:
            print("❌ Could not find video URL in page (fallback)")
            return None

        if video_src.startswith('//'):
            video_src = 'https:' + video_src
        elif video_src.startswith('/'):
            video_src = urljoin(page_url, video_src)

        if is_potentially_watermarked_url(video_src):
            cleaned = sanitize_video_url(video_src)
            if cleaned != video_src:
                print(f"🧼 Watermark flags detected. Using cleaned URL: {cleaned}")
                video_src = cleaned

        print(f"\n⬇️  Downloading video (HTTP fallback): {video_src}")
        try:
            with sess.get(video_src, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(output_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*256):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            print(f"❌ Download failed (fallback): {e}")
            return None

        file_size = os.path.getsize(output_filename) / (1024 * 1024)
        print(f"✅ Downloaded successfully!\n📁 Saved as: {output_filename}\n📊 Size: {file_size:.2f} MB")
        return output_filename

    # Try Playwright path first
    if not _HAS_PLAYWRIGHT:
        return _http_fallback(url)

    try:
        with sync_playwright() as p:
            print("🚀 Launching browser...")
            browser = p.chromium.launch(
                headless=False,
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-features=IsolateOrigins,site-per-process'
                ]
            )

            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                viewport={'width': 1920, 'height': 1080},
                locale='en-US',
                timezone_id='America/New_York'
            )

            # Load cookies if they exist
            if os.path.exists(cookies_file):
                print(f"📂 Loading cookies from {cookies_file}...")
                with open(cookies_file, 'r') as f:
                    cookies = json.load(f)
                    context.add_cookies(cookies)
                print("✅ Cookies loaded!")
            else:
                print("⚠️  No cookies found - run save_cookies.py first!")
                browser.close()
                return _http_fallback(url)

            page = context.new_page()

            # Hide webdriver
            page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)

            print(f"\n🌐 Navigating to: {url}")
            page.goto(url, wait_until='networkidle', timeout=60000)

            print("⏳ Waiting for page to load...")
            time.sleep(3)

            # Check if Cloudflare challenge appears
            if "cloudflare" in page.content().lower() or "just a moment" in page.content().lower():
                print("🔄 Cloudflare detected - waiting for it to resolve...")
                time.sleep(5)

            print("\n🔍 Looking for video element...")

            # Find video tag
            video_element = page.query_selector('video')

            if video_element:
                video_src = video_element.get_attribute('src')
                if not video_src:
                    source = page.query_selector('video source')
                    if source:
                        video_src = source.get_attribute('src')

                if video_src:
                    print(f"✅ Found video source: {video_src}")

                    # Prefer non-watermarked URL if watermark toggles are present
                    if is_potentially_watermarked_url(video_src):
                        cleaned = sanitize_video_url(video_src)
                        if cleaned != video_src:
                            print(f"🧼 Watermark flags detected in URL. Using cleaned URL: {cleaned}")
                            video_src = cleaned

                    # Make absolute URL if relative
                    if video_src.startswith('//'):
                        video_src = 'https:' + video_src
                    elif video_src.startswith('/'):
                        video_src = urljoin(url, video_src)

                    print(f"\n⬇️  Downloading video...")

                    response = page.request.get(video_src)

                    if response.ok:
                        with open(output_filename, 'wb') as f:
                            f.write(response.body())

                        file_size = os.path.getsize(output_filename) / (1024 * 1024)
                        print(f"✅ Downloaded successfully!")
                        print(f"📁 Saved as: {output_filename}")
                        print(f"📊 Size: {file_size:.2f} MB")

                        browser.close()
                        return output_filename
                    else:
                        print(f"❌ Download failed: HTTP {response.status}")
                else:
                    print("❌ Could not find video source URL")
            else:
                print("❌ No video element found on page")

            browser.close()
            # Try fallback if Playwright path didn't succeed
            return _http_fallback(url)
    except Exception as e:
        print(f"⚠️  Playwright path failed: {e}")
        return _http_fallback(url)


def remove_watermarks(input_video, output_video):
    """Remove watermarks from video using YOLOv8 + LaMa"""

    print("\n" + "=" * 60)
    print("STEP 2: Removing Watermarks")
    print("=" * 60)

    # Initialize
    print("\nInitializing models...")
    detector = YOLOWatermarkDetector()
    inpainter = LamaInpainter()

    # Open video
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS) if hasattr(cv2, 'cv') else cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH) if hasattr(cv2, 'cv') else cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT) if hasattr(cv2, 'cv') else cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT) if hasattr(cv2, 'cv') else cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo: {input_video}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    if not out.isOpened():
        print("❌ Failed to open video writer!")
        return False

    print(f"\n🚀 Processing video...")

    frames_processed = 0
    frames_with_watermark = 0
    last_valid_bbox = None
    global_watermark_position = None  # Cache position from early frames

    # Load template for fallback
    template_path = '../watermark_template.png'
    template = cv2.imread(template_path)
    has_template = template is not None

    # First pass: detect watermark position from first 10 frames
    print("🔍 Analyzing first frames to find watermark position...")
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES if hasattr(cv2, 'cv') else cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(min(10, total_frames)):
        ret, frame = cap.read()
        if ret:
            detections = detector.detect(frame, confidence_threshold=0.3, padding=30)
            if detections and detections[0]['confidence'] > 0.5:
                global_watermark_position = detections[0]['bbox']
                print(f"✅ Found watermark position: {global_watermark_position}")
                break

    # Reset to beginning
    cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES if hasattr(cv2, 'cv') else cv2.CAP_PROP_POS_FRAMES, 0)

    # Process frames
    for frame_num in tqdm(range(total_frames), desc="Processing"):
        ret, frame = cap.read()

        if not ret:
            break

        # Detect watermark
        detections = detector.detect(frame, confidence_threshold=0.3, padding=30)

        # Fallback 1: template matching (lower threshold)
        if not detections and has_template:
            th, tw = template.shape[:2]
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            if max_val > 0.5:  # Lowered from 0.6 to 0.5
                x1, y1 = max_loc
                x2, y2 = x1 + tw, y1 + th
                x1 = max(0, x1 - 30)
                y1 = max(0, y1 - 30)
                x2 = min(width, x2 + 30)
                y2 = min(height, y2 + 30)
                detections = [{'bbox': (x1, y1, x2, y2), 'confidence': max_val}]

        # Fallback 2: temporal consistency (last frame position)
        if not detections and last_valid_bbox:
            detections = [{'bbox': last_valid_bbox, 'confidence': 0.0}]

        # Fallback 3: global position (from first frames analysis)
        if not detections and global_watermark_position:
            detections = [{'bbox': global_watermark_position, 'confidence': 0.0}]

        if detections:
            frames_with_watermark += 1

            if detections[0]['confidence'] > 0.3:
                last_valid_bbox = detections[0]['bbox']

            mask = detector.create_mask(frame, detections)

            try:
                result = inpainter.inpaint_region(frame, mask)
                out.write(result)
            except Exception as e:
                out.write(frame)
        else:
            out.write(frame)

        frames_processed += 1

    cap.release()
    out.release()

    print("\n" + "=" * 60)
    print("✅ WATERMARK REMOVAL COMPLETE!")
    print("=" * 60)
    print(f"Frames processed: {frames_processed}")
    print(f"Watermarks detected: {frames_with_watermark}")
    print(f"📁 Saved to: {output_video}")

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Download Video + Remove Watermarks")
    print("=" * 60)

    url = input("\nEnter video URL: ").strip()

    # Step 1: Download
    print("\n" + "=" * 60)
    print("STEP 1: Downloading Video")
    print("=" * 60)

    downloaded_file = download_video(url, "downloaded_video.mp4")

    if downloaded_file:
        # Step 2: Remove watermarks
        output_file = "cleaned_video.mp4"
        success = remove_watermarks(downloaded_file, output_file)

        if success:
            print("\n" + "=" * 60)
            print("🎉 ALL DONE!")
            print("=" * 60)
            print(f"Original: {downloaded_file}")
            print(f"Cleaned: {output_file}")
    else:
        print("\n❌ Download failed - cannot proceed with watermark removal")
