@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo ============================================================
echo  INSERT VIDEO INTO IMAGE - Create Video-in-Frame Effect
echo ============================================================
echo.
echo This tool lets you:
echo   1. Select a background IMAGE (the wall/frame photo)
echo   2. Select a foreground VIDEO (to insert into the image)
echo   3. Draw a rectangle where the video should appear
echo   4. Output a new video with the video embedded in the image
echo.
echo ============================================================
echo.

REM === FILE PICKER FOR BACKGROUND IMAGE ===
echo [1/2] Select the BACKGROUND IMAGE (the photo with frame)...
set "PS_CMD=Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.OpenFileDialog; $f.Filter = 'Images (*.jpg;*.jpeg;*.png;*.bmp)|*.jpg;*.jpeg;*.png;*.bmp|All files (*.*)|*.*'; $f.Title = 'Select Background Image'; if($f.ShowDialog() -eq 'OK'){$f.FileName}"
for /f "delims=" %%I in ('powershell -NoProfile -Command "%PS_CMD%"') do set "BG_IMAGE=%%I"

if "%BG_IMAGE%"=="" (
    echo [ERROR] No image selected. Exiting.
    pause
    exit /b 1
)
echo [OK] Background image: %BG_IMAGE%
echo.

REM === FILE PICKER FOR FOREGROUND VIDEO ===
echo [2/2] Select the FOREGROUND VIDEO (to insert into the image)...
set "PS_CMD=Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.OpenFileDialog; $f.Filter = 'Videos (*.mp4;*.mov;*.avi;*.mkv)|*.mp4;*.mov;*.avi;*.mkv|All files (*.*)|*.*'; $f.Title = 'Select Foreground Video'; if($f.ShowDialog() -eq 'OK'){$f.FileName}"
for /f "delims=" %%I in ('powershell -NoProfile -Command "%PS_CMD%"') do set "FG_VIDEO=%%I"

if "%FG_VIDEO%"=="" (
    echo [ERROR] No video selected. Exiting.
    pause
    exit /b 1
)
echo [OK] Foreground video: %FG_VIDEO%
echo.

REM === RUN PYTHON SCRIPT ===
echo ============================================================
echo  TWO STEPS:
echo  1. Draw rectangle on VIDEO to CROP (remove black borders)
echo     - Press ESC to skip (use full video)
echo  2. Draw rectangle on IMAGE where video goes
echo.
echo  Controls:
echo  - Click and drag to draw the rectangle
echo  - Press ENTER to confirm
echo  - Press R to reset
echo  - Press ESC to cancel/skip
echo ============================================================
echo.

set PYTHON_PATH=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
if not exist "%PYTHON_PATH%" set PYTHON_PATH=python

"%PYTHON_PATH%" "%~dp0insert_video_into_image.py" "%BG_IMAGE%" "%FG_VIDEO%"

echo.
echo ============================================================
echo  DONE! Check the output folder for your video.
echo ============================================================
pause
