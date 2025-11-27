@echo off
cd /d D:\watermarkz
echo Sending first.mp4 to 4090 detection worker...
C:\Users\has\AppData\Local\Programs\Python\Python312\python.exe -c "from celery import Celery; c = Celery(broker='redis://default:bwQmxUCQEXUlYTWACmPbbkpnHPVpoiIa@tramway.proxy.rlwy.net:48930/0'); r = c.send_task('detection.detect_video', kwargs={'video_path': 'D:/watermarkz/videostotrain/first.mp4', 'mode': 'yolo'}, queue='detection'); print(f'Task sent! ID: {r.id}')"
pause
