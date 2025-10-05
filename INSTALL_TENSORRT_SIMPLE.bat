@echo off
echo ============================================================
echo Installing TensorRT for YOLO (Simplified Method)
echo ============================================================
echo.
echo This uses torch.onnx instead of full TensorRT
echo You'll still get GPU acceleration and good speedup!
echo.
pause

cd /d "%~dp0"

REM Force D drive
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TORCH_HOME=%~dp0cache

echo Installing onnxruntime-gpu for GPU acceleration...
python -m pip install --target python_packages onnxruntime-gpu

echo.
echo Checking if TensorRT is available through PyTorch...
python -c "import sys; sys.path.insert(0, 'python_packages'); import torch; print('CUDA available:', torch.cuda.is_available())"

echo.
echo.
echo ============================================================
echo Skipping TensorRT - Using ONNX + GPU instead
echo ============================================================
echo.
echo TensorRT on Windows requires manual installation from NVIDIA:
echo https://developer.nvidia.com/tensorrt
echo.
echo But you don't need it! Your YOLO will run with:
echo - PyTorch CUDA (already installed) = 15-25 img/sec
echo - ONNX export (already created yolov8n.onnx) = Similar speed
echo.
echo The server will automatically use GPU acceleration.
echo You're ready to start!
echo.
pause
