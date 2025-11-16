@echo off
cd /d "%~dp0"

echo ============================================================
echo PROCESS VIDEO WITH SAM2 MASKS (WSL2)
echo ============================================================
echo.
echo This will:
echo   1. Let you pick a video file
echo   2. Use existing masks from temp_sam2_masks/
echo   3. Process with ProPainter in WSL2
echo   4. Save result to results/
echo.
echo ============================================================
echo.

REM Use PowerShell to show file picker
echo Opening file picker...
echo.

powershell -ExecutionPolicy Bypass -Command ^
"Add-Type -AssemblyName System.Windows.Forms; ^
$dialog = New-Object System.Windows.Forms.OpenFileDialog; ^
$dialog.Title = 'Select Video to Process with SAM2 Masks'; ^
$dialog.Filter = 'Video Files (*.mp4;*.avi;*.mov;*.mkv)|*.mp4;*.avi;*.mov;*.mkv|All Files (*.*)|*.*'; ^
$dialog.InitialDirectory = 'D:\watermarkz\videostotrain'; ^
if ($dialog.ShowDialog() -eq 'OK') { ^
    $dialog.FileName ^
} else { ^
    exit 1 ^
}" > "%TEMP%\selected_video.txt"

if %errorlevel% neq 0 (
    echo No video selected!
    pause
    exit /b 1
)

set /p VIDEO_PATH=<"%TEMP%\selected_video.txt"
del "%TEMP%\selected_video.txt"

echo Selected: %VIDEO_PATH%
echo.
echo ============================================================
echo.

REM Run the PowerShell script with selected video
powershell -ExecutionPolicy Bypass -File "%~dp0RUN_SAM2_WSL2_WITH_MASKS.ps1" -VideoPath "%VIDEO_PATH%"

echo.
echo ============================================================
echo DONE!
echo ============================================================
pause
