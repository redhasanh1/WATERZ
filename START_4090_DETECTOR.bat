@echo off
cd /d D:\watermarkz

echo ========================================
echo   4090 Detection Worker - Send Task
echo ========================================
echo.
echo  [1] YOLO Detection (automatic)
echo  [2] SAM2 Detection (interactive)
echo.
choice /c 12 /n /m "Select mode (1 or 2): "

if errorlevel 2 (
    set "MODE=sam2"
    set "MODE_NAME=SAM2 Interactive"
) else (
    set "MODE=yolo"
    set "MODE_NAME=YOLO Automatic"
)

echo.
echo Selected: %MODE_NAME%
echo.

:: Open file picker dialog using PowerShell
echo Select a video file...
for /f "delims=" %%I in ('powershell -command "& {Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.OpenFileDialog; $f.InitialDirectory = 'D:\watermarkz\videostotrain'; $f.Filter = 'Video Files (*.mp4;*.avi;*.mkv;*.mov)|*.mp4;*.avi;*.mkv;*.mov|All Files (*.*)|*.*'; $f.Title = 'Select Video for 4090 Detection'; if ($f.ShowDialog() -eq 'OK') { $f.FileName } else { 'CANCELLED' }}"') do set "VIDEO_PATH=%%I"

if "%VIDEO_PATH%"=="CANCELLED" (
    echo File selection cancelled.
    pause
    exit /b 0
)

if "%VIDEO_PATH%"=="" (
    echo No file selected.
    pause
    exit /b 1
)

:: Convert backslashes to forward slashes for Python
set "VIDEO_PATH_UNIX=%VIDEO_PATH:\=/%"

echo.
echo Sending: %VIDEO_PATH%
echo Mode: %MODE_NAME%
echo To 4090 detection worker...
echo.

C:\Users\has\AppData\Local\Programs\Python\Python312\python.exe -c "from celery import Celery; c = Celery(broker='redis://default:bwQmxUCQEXUlYTWACmPbbkpnHPVpoiIa@tramway.proxy.rlwy.net:48930/0'); r = c.send_task('detection.detect_video', kwargs={'video_path': '%VIDEO_PATH_UNIX%', 'mode': '%MODE%'}, queue='detection'); print(f'Task sent! ID: {r.id}')"
pause
