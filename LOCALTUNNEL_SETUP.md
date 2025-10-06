# Localtunnel Setup Guide

## Why Localtunnel?
- ✅ **No "Visit Site" warning** - Users connect directly
- ✅ **Auto URL detection** - No manual config.js updates
- ✅ **Free forever** - No paid plans needed
- ✅ **Everything in watermarkz folder** - No C drive usage
- ✅ **Easy setup** - Just 2 steps

---

## Setup (One-Time)

### Step 1: Install Localtunnel
```batch
INSTALL_LOCALTUNNEL.bat
```

This will:
- Download portable Node.js to `watermarkz/node/`
- Install localtunnel to `watermarkz/node_modules/`
- All files stay in watermarkz folder

**Time:** ~2-3 minutes (depending on internet speed)

---

### Step 2: Start Everything
```batch
START_ALL.bat
```

This will:
1. Start Localtunnel and capture public URL
2. Start Redis (job queue)
3. Start Celery (background worker)
4. Start Flask (web server)

**Check the START_ALL window** - it will show your public URL!

---

## How It Works

### Backend (Auto URL Capture)
1. `START_LOCALTUNNEL.bat` launches tunnel
2. Captures URL like `https://cool-dog-1234.loca.lt`
3. Writes to `web/tunnel_url.txt`
4. Server serves this file at `/tunnel_url.txt`

### Frontend (Auto URL Detection)
1. `web/config.js` fetches `/tunnel_url.txt` on page load
2. Sets `API_BASE_URL` automatically
3. All API calls use the tunnel URL
4. Fallback to localhost if tunnel not found

**Result:** No manual config updates needed!

---

## File Locations

All files stay in watermarkz:
```
watermarkz/
├── node/                   (Portable Node.js)
├── node_modules/           (Localtunnel)
├── node_cache/             (npm cache)
├── tunnel_output.txt       (Tunnel logs)
└── web/tunnel_url.txt      (Auto-generated URL)
```

**C drive usage:** 0 bytes

---

## Troubleshooting

### Tunnel URL Not Showing
Check `tunnel_output.txt` for errors:
```batch
type tunnel_output.txt
```

### Frontend Not Connecting
1. Check browser console (F12)
2. Look for "Connected to tunnel" message
3. If shows localhost, tunnel_url.txt wasn't found

### Restart Tunnel
Close Localtunnel window and run:
```batch
START_LOCALTUNNEL.bat
```

New URL will be auto-detected!

---

## vs ngrok

| Feature | Localtunnel | ngrok Free |
|---------|------------|-----------|
| Warning Screen | ❌ None | ⚠️ "Visit Site" button |
| URL Changes | On restart | On restart |
| Auto-Detection | ✅ Yes | ❌ Manual |
| C Drive Usage | 0 MB | ~50 MB |
| Setup | 1 command | Login + config |

---

## Need Custom Domain?

For `markremoverai.com` → tunnel URL:

**Option 1:** ngrok paid ($8/month)
**Option 2:** Cloudflare Tunnel (FREE, 10 min setup)
**Option 3:** VPS reverse proxy ($5/month)

Current setup: Users see your domain, backend uses tunnel URL
