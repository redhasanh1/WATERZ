from playwright.sync_api import sync_playwright
import time
import os
import json

def download_video_with_cookies(url, output_filename="downloaded_video.mp4", cookies_file="cookies.json"):
    """
    Download video using Playwright with saved cookies

    Args:
        url: Direct video URL to download
        output_filename: Name to save the video as
        cookies_file: Path to cookies JSON file (optional)
    """

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


if __name__ == "__main__":
    print("=" * 60)
    print("Video Downloader with Cookie Authentication")
    print("=" * 60)

    url = input("\nEnter video URL: ").strip()
    output_name = input("Enter output filename (default: downloaded_video.mp4): ").strip()

    if not output_name:
        output_name = "downloaded_video.mp4"

    download_video_with_cookies(url, output_name)
