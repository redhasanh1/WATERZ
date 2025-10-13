import sys
import os
import time
import json

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
from url_utils import sanitize_video_url, is_potentially_watermarked_url
import re
import html
from urllib.parse import urljoin

def download_video_with_cookies(url, output_filename="downloaded_video.mp4", cookies_file="cookies.json"):
    """
    Download video using Playwright with saved cookies.
    Falls back to HTTP+cookies if Playwright is unavailable or video element isn't accessible.
    """

    def _http_fallback(page_url: str) -> bool:
        print("\n🛟 Falling back to HTTP session scraping...")
        import requests

        if not os.path.exists(cookies_file):
            print("⚠️  No cookies found - will proceed without authentication")
            sess = requests.Session()
        else:
            sess = requests.Session()
            sess.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
            with open(cookies_file, 'r') as f:
                cookies = json.load(f)
                for c in cookies:
                    domain = c.get('domain') or ''
                    sess.cookies.set(
                        name=c.get('name'),
                        value=c.get('value'),
                        domain=domain or None,
                        path=c.get('path', '/'),
                    )

        try:
            resp = sess.get(page_url, timeout=30)
            if resp.status_code >= 400:
                print(f"❌ HTTP error fetching page: {resp.status_code}")
                return False
            content = resp.text
        except Exception as e:
            print(f"❌ HTTP error: {e}")
            return False

        mp4_urls = re.findall(r'https?://[^\s"\'<>]+\.mp4[^\s"\'<>]*', content)
        video_src = None
        if mp4_urls:
            video_src = html.unescape(mp4_urls[0])
        else:
            m = re.search(r'<video[^>]+src=["\']([^"\']+)["\']', content, re.IGNORECASE)
            if m:
                video_src = html.unescape(m.group(1))

        if not video_src:
            print("❌ Could not find video URL in page (fallback)")
            return False

        if video_src.startswith('//'):
            video_src = 'https:' + video_src
        elif video_src.startswith('/'):
            video_src = urljoin(page_url, video_src)

        if is_potentially_watermarked_url(video_src):
            cleaned = sanitize_video_url(video_src)
            if cleaned != video_src:
                print(f"🧼 Watermark flags detected. Using cleaned URL: {cleaned}")
                video_src = cleaned

        print(f"\n⬇️  Downloading video (HTTP fallback)...")
        try:
            with sess.get(video_src, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(output_filename, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*256):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            print(f"❌ Download failed (fallback): {e}")
            return False

        file_size = os.path.getsize(output_filename) / (1024 * 1024)
        print(f"✅ Downloaded successfully!\n📁 Saved as: {output_filename}\n📊 Size: {file_size:.2f} MB")
        return True

    if not _HAS_PLAYWRIGHT:
        if _http_fallback(url):
            return
        print("❌ Fallback also failed.")
        return

    try:
        with sync_playwright() as p:
        print("🚀 Launching browser...")
        browser = p.chromium.launch(headless=False)  # Set to True for headless
        context = browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )

        # Load cookies if they exist
        if os.path.exists(cookies_file):
            print(f"📂 Loading cookies from {cookies_file}...")
            with open(cookies_file, 'r') as f:
                cookies = json.load(f)
                context.add_cookies(cookies)
            print("✅ Cookies loaded!")
        else:
            print("⚠️  No cookies found - will proceed without authentication")

        page = context.new_page()

        print(f"\n🌐 Navigating to: {url}")

        # Navigate to the URL
        page.goto(url, wait_until='networkidle', timeout=60000)

        print("⏳ Waiting for page to load...")
        time.sleep(3)

        # Check if Cloudflare challenge appears
        if "cloudflare" in page.content().lower() or "just a moment" in page.content().lower():
            print("🔄 Cloudflare detected - waiting for it to resolve...")
            time.sleep(5)

        # Try to find and download video
        print("\n🔍 Looking for video element...")

        # Method 1: Find video tag
        video_element = page.query_selector('video')

        if video_element:
            video_src = video_element.get_attribute('src')
            if not video_src:
                # Try source tag inside video
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
                    from urllib.parse import urljoin
                    video_src = urljoin(url, video_src)

                print(f"\n⬇️  Downloading video...")

                # Download using Playwright's request context (preserves cookies/auth)
                response = page.request.get(video_src)

                if response.ok:
                    with open(output_filename, 'wb') as f:
                        f.write(response.body())

                    file_size = os.path.getsize(output_filename) / (1024 * 1024)  # MB
                    print(f"✅ Downloaded successfully!")
                    print(f"📁 Saved as: {output_filename}")
                    print(f"📊 Size: {file_size:.2f} MB")
                else:
                    print(f"❌ Download failed: HTTP {response.status}")
            else:
                print("❌ Could not find video source URL")
        else:
            print("❌ No video element found on page")
            print("\n💡 Try using the 'Save Cookies' script first to authenticate")

        browser.close()
        print("\n✅ Done!")
    except Exception as e:
        print(f"⚠️  Playwright path failed: {e}")
        if _http_fallback(url):
            return
        print("❌ Fallback also failed.")


if __name__ == "__main__":
    print("=" * 60)
    print("Video Downloader with Cookie Authentication")
    print("=" * 60)

    url = input("\nEnter video URL: ").strip()
    output_name = input("Enter output filename (default: downloaded_video.mp4): ").strip()

    if not output_name:
        output_name = "downloaded_video.mp4"

    download_video_with_cookies(url, output_name)
