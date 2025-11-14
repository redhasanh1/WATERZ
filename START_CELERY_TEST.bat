@echo off
cd /d D:\watermarkz
set FORCE_TRT_RAFT=0
set USE_NEUFLOW=1
set ENABLE_FLASH_ATTENTION=1
set SEGMENT_WORKERS=4
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -m celery -A server_production.celery worker --loglevel=info --pool=threads --concurrency=4
