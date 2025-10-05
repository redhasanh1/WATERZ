# Video Downloader with Cookie Authentication

Download videos from websites that require login, bypassing Cloudflare protection.

## Setup

1. Install Playwright:
```bash
pip install playwright
playwright install chromium
```

## Usage

### Step 1: Save Cookies (One Time)

Run this to login and save your session:

```bash
python save_cookies.py
```

1. Browser will open
2. Login to the website
3. Press ENTER in terminal
4. Cookies saved to `cookies.json`

### Step 2: Download Videos

Now you can download videos without logging in again:

```bash
python download_video.py
```

Enter the video URL and it will download using your saved cookies!

## How It Works

- **Playwright**: Uses real Chrome browser (bypasses Cloudflare easily)
- **Cookies**: Saves your login session so you don't need to login again
- **Headless**: Can run in background (set `headless=True` in code)

## Files

- `save_cookies.py` - Login once and save cookies
- `download_video.py` - Download videos using saved cookies
- `cookies.json` - Your saved login session (created after first run)
