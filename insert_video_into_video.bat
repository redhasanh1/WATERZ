@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo ============================================================
echo  INSERT VIDEO INTO VIDEO - Overlay Video on Video
echo ============================================================
echo.
echo This tool lets you:
echo   1. Select a BACKGROUND VIDEO
echo   2. Select a FOREGROUND VIDEO (to overlay)
echo   3. Draw a rectangle on foreground to CROP (optional)
echo   4. Draw a rectangle on background where to place it
echo   5. Output a new video with the overlay
echo.
echo ============================================================
echo.

REM === FILE PICKER FOR BACKGROUND VIDEO ===
echo [1/2] Select the BACKGROUND VIDEO...
set "PS_CMD=Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.OpenFileDialog; $f.Filter = 'Videos (*.mp4;*.mov;*.avi;*.mkv;*.webm)|*.mp4;*.mov;*.avi;*.mkv;*.webm|All files (*.*)|*.*'; $f.Title = 'Select Background Video'; if($f.ShowDialog() -eq 'OK'){$f.FileName}"
for /f "delims=" %%I in ('powershell -NoProfile -Command "%PS_CMD%"') do set "BG_VIDEO=%%I"

if "%BG_VIDEO%"=="" (
    echo [ERROR] No video selected. Exiting.
    pause
    exit /b 1
)
echo [OK] Background video: %BG_VIDEO%
echo.

REM === FILE PICKER FOR FOREGROUND VIDEO ===
echo [2/2] Select the FOREGROUND VIDEO (to overlay)...
set "PS_CMD=Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.OpenFileDialog; $f.Filter = 'Videos (*.mp4;*.mov;*.avi;*.mkv;*.webm)|*.mp4;*.mov;*.avi;*.mkv;*.webm|All files (*.*)|*.*'; $f.Title = 'Select Foreground Video'; if($f.ShowDialog() -eq 'OK'){$f.FileName}"
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
echo  1. Draw rectangle on FOREGROUND to CROP (remove borders)
echo     - Press ESC to skip (use full video)
echo  2. Draw rectangle on BACKGROUND where overlay goes
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

"%PYTHON_PATH%" "%~dp0insert_video_into_video.py" "%BG_VIDEO%" "%FG_VIDEO%"

echo.
echo ============================================================
echo  DONE! Check the output folder for your video.
echo ============================================================
pause
