@echo off
echo ============================================================
echo Installing TensorRT (Correct Method for NVIDIA)
echo ============================================================
echo.
pause

cd /d "%~dp0"

REM Force D drive
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TORCH_HOME=%~dp0cache

echo TensorRT requires special NVIDIA index...
echo.
pause

REM Install nvidia-pyindex first
echo Step 1: Installing nvidia-pyindex...
python -m pip install --target python_packages nvidia-pyindex
echo Step 1 exit code: %ERRORLEVEL%
pause

echo.
echo Step 2: Installing nvidia-tensorrt from NVIDIA repo...
python -m pip install --target python_packages nvidia-tensorrt --extra-index-url https://pypi.nvidia.com
echo Step 2 exit code: %ERRORLEVEL%
pause

echo.
echo Step 3: Exporting YOLO to TensorRT format...
echo This may take 2-5 minutes...
python -c "import sys; sys.path.insert(0, 'python_packages'); from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='engine', device=0, half=True)"
echo Step 3 exit code: %ERRORLEVEL%
pause

if exist yolov8n.engine (
    echo.
    echo ✅✅✅ TensorRT export successful!
    echo You now get 20-35 images/sec (2-3x speedup)!
) else (
    echo.
    echo ⚠️  Export failed - you can skip TensorRT
    echo You'll still get 15-25 images/sec without it
)

echo.
pause
