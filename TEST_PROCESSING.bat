@echo off
cd /d "%~dp0"

REM Force D drive temp/cache
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TMPDIR=%~dp0temp
set TORCH_HOME=%~dp0cache
set XDG_CACHE_HOME=%~dp0cache
set OPENCV_TEMP_PATH=%~dp0temp
set PYTHONPATH=%~dp0python_packages

REM Add TensorRT DLLs to PATH
set PATH=%~dp0python_packages;%~dp0python_packages\torch\lib;%~dp0TensorRT-10.7.0.23\lib;%PATH%

echo ======================================
echo Testing Video Processing with TensorRT
echo ======================================
echo.

REM Check if test video exists
if not exist "uploads\*.mp4" (
    echo ❌ No videos found in uploads folder!
    echo.
    echo Please put a test video in the uploads\ folder first.
    pause
    exit /b
)

REM Find first video in uploads
for %%f in (uploads\*.mp4) do (
    set TEST_VIDEO=%%f
    goto :found
)

:found
echo Testing with: %TEST_VIDEO%
echo.

python test_video_processing.py "%TEST_VIDEO%"

echo.
echo ======================================
echo Check results\ folder for output!
echo ======================================
pause
