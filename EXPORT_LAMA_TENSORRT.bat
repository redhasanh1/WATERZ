@echo off
cd /d "%~dp0"

set PYTHONPATH=%~dp0python_packages
set PATH=%~dp0TensorRT-10.7.0.23\bin;%~dp0python_packages;%~dp0python_packages\torch\lib;%~dp0TensorRT-10.7.0.23\lib;%PATH%

echo ========================================
echo LAMA INPAINTING TENSORRT EXPORT
echo ========================================
echo.
echo This will:
echo   1. Load LAMA PyTorch model
echo   2. Export to ONNX format
echo   3. Convert to TensorRT engine (FP16)
echo.
echo Expected result:
echo   - 2-5x faster inpainting
echo   - Same quality as original LAMA
echo.
echo This will take 5-10 minutes...
echo ========================================
echo.

python export_lama_tensorrt.py
pause
