# 🚀 Railway Auto-Upload Setup Guide

## What This Does

Automatically uploads processed videos from your **LOCAL GPU worker** to **Railway cloud server** so users can download at maximum speed!

```
LOCAL WORKER (GPU) → Process Video → Upload to Railway → USER DOWNLOADS (FAST!)
```

## Quick Setup (2 Steps!)

### ✅ Step 1: Ensure tunnel_url.txt exists

The file `web/tunnel_url.txt` should contain your Railway/production URL:

```bash
# File: web/tunnel_url.txt
https://markremoverai.com
```

**You already have this!** ✅

### ✅ Step 2: Restart Celery Worker

Simply restart your Celery worker to load the new code:

```bash
# Close existing Celery window (Ctrl+C)
# Then run again:
START_CELERY.bat
```

You should see:
```
[UPLOAD] Loaded TUNNEL_URL from web\tunnel_url.txt: https://markremoverai.com
[UPLOAD] ✅ Auto-upload to Railway ENABLED
```

## How to Verify It's Working

After processing a video, check Celery logs for:

### ✅ SUCCESS - Upload Working:
```
[FINALIZE] ✓ Final video ready: D:\watermarkz\results\video_propainter.mp4 (3.76 MB)
[FINALIZE] Upload config - TUNNEL_URL: SET, API_BASE_URL: NOT SET, UPLOAD_RESULT_BACK: 1
[FINALIZE] 📤 Uploading result to Railway: https://markremoverai.com/api/upload-result
[FINALIZE] ✅ Result uploaded to Railway: /results/video_propainter.mp4
```

### ❌ NOT WORKING - Missing Config:
```
[FINALIZE] ✓ Final video ready: D:\watermarkz\results\video_propainter.mp4 (3.76 MB)
[FINALIZE] Upload config - TUNNEL_URL: NOT SET, API_BASE_URL: NOT SET, UPLOAD_RESULT_BACK: 1
[FINALIZE] ⚠️  Skipping Railway upload - TUNNEL_URL/API_BASE_URL not set
```

## Advanced Configuration

### Disable Auto-Upload (Optional)

If you want to disable auto-upload temporarily:

```bash
set UPLOAD_RESULT_BACK=0
START_CELERY.bat
```

### Use Different URL (Optional)

If you want to use a different URL than `web/tunnel_url.txt`:

```bash
set TUNNEL_URL=https://your-custom-url.com
START_CELERY.bat
```

## Troubleshooting

### Upload fails with timeout
- **Cause**: Large video or slow connection
- **Fix**: Already handled - timeout is 300s (5 minutes)

### Upload fails with 404 error
- **Cause**: Railway server not running or wrong URL
- **Fix**: Check Railway deployment, verify URL in `web/tunnel_url.txt`

### Upload fails with connection error
- **Cause**: Network issue or Railway down
- **Fix**: Videos still work locally! User can download from local worker (fallback)

## How It Works

1. **Worker finishes** → Video saved to `D:\watermarkz\results\`
2. **Auto-detect** → Reads `TUNNEL_URL` from environment
3. **Upload** → POSTs video to `https://markremoverai.com/api/upload-result`
4. **Railway saves** → Stores in `/data/results/` (Railway volume)
5. **Redis updated** → Path changed from local to Railway: `/results/filename`
6. **User downloads** → From Railway CDN (SUPER FAST!)

## Benefits

- ✅ **Faster downloads** - Railway CDN edge locations
- ✅ **Worker freed** - No need to serve downloads
- ✅ **Parallel processing** - Upload happens async
- ✅ **Graceful fallback** - Works even if upload fails
- ✅ **Auto cleanup** - Railway deletes after download

---

**Need help?** Check Celery logs for detailed diagnostic messages!
