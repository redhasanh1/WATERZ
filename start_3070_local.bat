@echo off
cd /d D:\watermarkz
set PYTHONPATH=D:\watermarkz\faster-propainter-main
set REDIS_URL=redis://192.168.0.105:6379/0
C:\Users\has\AppData\Local\Programs\Python\Python312\python.exe -m celery -A celery_3070_worker worker --loglevel=info --pool=solo -Q propainter -n propainter@%COMPUTERNAME%
