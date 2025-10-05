@echo off
echo ============================================================
echo Upgrading to CUDA 12.1 + PyTorch + TensorRT 10 + cuDNN 9
echo ============================================================
echo.
echo This will:
echo   1. Uninstall PyTorch CUDA 11.8
echo   2. Install PyTorch CUDA 12.1
echo   3. Prepare for TensorRT 10.x installation
echo.
echo Your GPU: GTX 1660 Ti supports CUDA 12.6 ✅
echo.
pause

cd /d "%~dp0"

REM Force D drive
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TORCH_HOME=%~dp0cache

echo.
echo ============================================================
echo Step 1: Uninstalling old PyTorch (CUDA 11.8)
echo ============================================================
echo.

python -m pip uninstall -y torch torchvision torchaudio

echo.
echo ============================================================
echo Step 2: Installing PyTorch with CUDA 12.1
echo ============================================================
echo.
echo Downloading from PyTorch repository...
echo This may take 5-10 minutes (large files)
echo.

python -m pip install --cache-dir pip_cache --target python_packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo ============================================================
echo Step 3: Testing PyTorch CUDA 12.1
echo ============================================================
echo.

python -c "import sys; sys.path.insert(0, 'python_packages'); import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('CUDA Version:', torch.version.cuda)"

echo.
echo ============================================================
echo PyTorch CUDA 12.1 Installation Complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Download TensorRT 10.x for CUDA 12.x from NVIDIA
echo   2. Download cuDNN 9.x for CUDA 12.x from NVIDIA
echo.
echo I'll create instructions for the downloads...
echo.
pause
