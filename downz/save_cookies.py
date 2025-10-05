from playwright.sync_api import sync_playwright
import json
import time

def save_cookies_after_login(website_url, cookies_file="cookies.json"):
    """
    Opens browser, lets you login manually, then saves cookies

    Args:
        website_url: URL to login to
        cookies_file: Where to save cookies
    """

    with sync_playwright() as p:
        print("🚀 Launching browser...")
        print("⚠️  Browser will open - please login manually")
        print("⚠️  After logging in, press ENTER in this terminal to save cookies\n")

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

        # Hide webdriver property
        page = context.new_page()
        page.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)

        print(f"🌐 Navigating to: {website_url}")
        page.goto(website_url, wait_until='domcontentloaded', timeout=60000)

        # Wait for Cloudflare to resolve
        print("⏳ Waiting for Cloudflare check...")
        time.sleep(5)

        # Wait for user to login
        input("\n👉 Press ENTER after you've logged in...")

        # Save cookies
        cookies = context.cookies()

        with open(cookies_file, 'w') as f:
            json.dump(cookies, f, indent=2)

        print(f"\n✅ Saved {len(cookies)} cookies to {cookies_file}")
        print("You can now use download_video.py with authentication!")

        browser.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Cookie Saver - Login Once, Use Forever")
    print("=" * 60)

    url = input("\nEnter website URL to login: ").strip()

    save_cookies_after_login(url)
