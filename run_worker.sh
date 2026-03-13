#!/bin/bash
docker rm -f watermarkz-worker
docker run -it --gpus all \
  --name watermarkz-worker \
  --read-only \
  --cap-drop=ALL \
  --cap-add=SYS_NICE \
  --security-opt no-new-privileges:true \
  --tmpfs /tmp:rw,exec,size=4g \
  --tmpfs /app/temp:rw,exec,size=4g \
  --tmpfs /app/uploads:rw,exec,size=4g \
  --tmpfs /app/results:rw,exec,size=4g \
  --tmpfs /app/cache:rw,exec,size=2g \
  --memory=20g \
  --cpus=8 \
  --pids-limit=512 \
  -e REDIS_URL="redis://default:bwQmxUCQEXUlYTWACmPbbkpnHPVpoiIa@tramway.proxy.rlwy.net:48930" \
  -e B2_KEY_ID="00539db5c1104b50000000003" \
  -e B2_APP_KEY="K005384b8lPoBT11wScxkZ2Gx0fszus" \
  -e B2_BUCKET="watermarkz" \
  -e NOTIFY_WEBHOOK_URL="https://ntfy.sh/watermarkz-workers-9f31c" \
  -e NOTIFY_WORKER_NAME="Local-4090" \
  humblewoslayer/watermarkz-celery:v12 \
  --concurrency=1
