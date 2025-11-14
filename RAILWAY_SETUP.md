# Railway Deployment Setup Guide

This guide walks you through deploying the watermark removal service to Railway, eliminating ngrok costs entirely.

## Architecture Overview

```
┌─────────────────────────────────────────┐
│         RAILWAY (Cloud)                 │
│  ┌──────────────────────────────────┐  │
│  │   Flask API (server_production)  │  │
│  │   - Receives upload requests     │  │
│  │   - Stores videos in /data       │  │
│  │   - Sends tasks to Celery        │  │
│  │   - Serves processed results     │  │
│  └──────────────────────────────────┘  │
│  ┌──────────────────────────────────┐  │
│  │   Redis (Task Queue)             │  │
│  │   - Managed by Railway           │  │
│  │   - Persistent task storage      │  │
│  └──────────────────────────────────┘  │
│  ┌──────────────────────────────────┐  │
│  │   Persistent Volume (100GB)      │  │
│  │   - Mounted at /data             │  │
│  │   - Stores uploads & results     │  │
│  └──────────────────────────────────┘  │
│  ┌──────────────────────────────────┐  │
│  │   Static Website (frontend)      │  │
│  │   - markremoverai.com            │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
                    ↕
        (Downloads/uploads videos
         via Flask API endpoints)
                    ↕
┌─────────────────────────────────────────┐
│      LOCAL PC (Windows + RTX 4090)      │
│  ┌──────────────────────────────────┐  │
│  │  Celery Workers (4 concurrent)   │  │
│  │  - Downloads from Railway API    │  │
│  │  - Processes with TensorRT       │  │
│  │  - Uploads results to Railway    │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## Step 1: Railway Setup

### 1.1 Create Railway Redis Service
1. Go to https://railway.app/dashboard
2. Click on your existing project (where markremoverai.com is hosted)
3. Click "+ New Service" → "Database" → "Add Redis"
4. Railway will provision a managed Redis instance

### 1.2 Get Redis Connection URL
1. Click on the Redis service
2. Go to "Variables" tab
3. Copy the `REDIS_URL` value (looks like: `redis://default:password@redis.railway.internal:6379`)

### 1.3 Save Redis URL Locally
Update `D:\watermarkz\redis_url.txt`:
```
redis://default:password@redis.railway.internal:6379
```
(Replace with your actual Railway Redis URL)

### 1.4 Deploy Flask API to Railway

#### Option A: Deploy via GitHub (Recommended)
1. Push code to GitHub:
   ```bash
   git add Procfile railway.toml requirements.txt .gitignore
   git commit -m "Railway deployment: Flask API + Redis + Persistent Volumes"
   git push origin main
   ```

2. In Railway dashboard:
   - Click "+ New Service" → "GitHub Repo"
   - Select your watermarkz repository
   - Railway will auto-detect Procfile and deploy
   - Railway will automatically create the /data volume (configured in railway.toml)

#### Option B: Deploy via Railway CLI
1. Install Railway CLI:
   ```bash
   npm install -g @railway/cli
   ```

2. Login and deploy:
   ```bash
   railway login
   railway link  # Select your existing project
   railway up
   ```

### 1.5 Configure Environment Variables in Railway
1. Click on your Flask API service
2. Go to "Variables" tab
3. Add the following variables:

| Variable Name           | Value                                    |
|------------------------|------------------------------------------|
| `REDIS_URL`            | (Copy from Railway Redis service)        |
| `CELERY_BROKER_URL`    | (Same as REDIS_URL)                      |
| `CELERY_RESULT_BACKEND`| (Same as REDIS_URL)                      |
| `SECRET_KEY`           | (Generate random 32-char string)         |

Generate SECRET_KEY:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 1.6 Verify Deployment
1. Check deployment logs in Railway dashboard
2. Your API should be available at: `https://your-project-name.railway.app`
3. Test health endpoint: `https://your-project-name.railway.app/health`

## Step 2: Local Worker Setup

### 2.1 Start Local Celery Workers
On your Windows PC with RTX 4090, run:
```batch
START_CELERY_RAILWAY.bat
```

This will:
- Load Railway Redis URL from `redis_url.txt`
- Start 4 concurrent Celery workers with TensorRT optimizations
- Connect to Railway Redis for task processing

### 2.2 Verify Worker Connection
You should see in the console:
```
[REDIS] Loaded Railway URL: redis://default:...@redis.railway.internal:6379
✅ Visual Studio C++ environment activated
[TensorRT] Environment configured
Starting Celery workers with TensorRT optimizations...
celery@DESKTOP-XXX ready.
```

## Step 3: Update Frontend

### 3.1 Point Frontend to Railway API
Update your static website (`markremoverai.com`) to use the new Railway API URL instead of ngrok:

```javascript
// OLD (ngrok):
const API_URL = 'https://abc123.ngrok-free.app';

// NEW (Railway):
const API_URL = 'https://your-project-name.railway.app';
```

### 3.2 Test Upload Flow
1. Upload a video through your website
2. Check Railway API logs: Video should be saved to /data/uploads
3. Check local Celery logs: Worker should download from Railway, process with TensorRT, upload result
4. Download result through Railway API

## Architecture Flow

### Upload Request:
```
User → Frontend (markremoverai.com)
     → Railway Flask API
     → Save to /data/uploads (Railway Volume)
     → Send Celery task (via Railway Redis)
     → Return task_id to user
```

### Processing:
```
Local Worker (RTX 4090)
  → Poll Railway Redis for tasks
  → Download video from Railway API (/download/<task_id>)
  → Process with TensorRT (YOLO, NeuFlow, RFCNet)
  → Upload result to Railway API (/upload_result/<task_id>)
  → Update task status in Redis
```

### Download Request:
```
User → Frontend
     → Check task status (Railway API)
     → Get download URL (Railway API /results/<task_id>)
     → Download from Railway Volume
```

## Cost Breakdown

| Service             | Cost                              |
|---------------------|-----------------------------------|
| Railway Plan        | **Already paying** (8GB/8 vCPU)   |
| Railway Redis       | **Included** in plan              |
| Railway Volumes     | **Included** (100GB shared disk)  |
| ngrok               | **$0** (eliminated!)              |
| **Total**           | **$0 additional cost**            |

## Troubleshooting

### Workers Can't Connect to Railway Redis
- Check `redis_url.txt` has correct Railway Redis URL
- Verify Railway Redis is running (check dashboard)
- Make sure no firewall blocks Redis port

### Volume Not Persisting Data
- Verify railway.toml has `[[deploy.volumes]]` section
- Check Railway dashboard shows volume mounted at /data
- Redeploy if volume was added after initial deployment

### Deployment Fails on Railway
- Check `requirements.txt` is committed
- Verify Procfile exists and is correct
- Check Railway logs for error details
- Ensure railway.toml is in repository root

### Out of Disk Space
- Railway plan includes 100GB shared disk across all services
- Check current usage in Railway dashboard
- Delete old videos from /data/uploads and /data/results
- Consider adding cleanup cronjob to auto-delete videos older than 7 days

## Next Steps

1. Monitor Railway metrics (CPU, RAM, disk usage, bandwidth)
2. Add automated cleanup for old videos (e.g., delete after 7 days)
3. Set up monitoring/alerts for failed tasks
4. Consider adding webhook notifications when processing completes

## Performance Tips

- Railway Volumes are fast for small files but slower for large videos
- If you process many large videos daily, S3 might be faster (optional upgrade)
- Current setup is perfect for moderate usage (< 50 videos/day)
- Workers can process 4 videos concurrently with RTX 4090

---

**Need help?** Check Railway docs: https://docs.railway.app/
