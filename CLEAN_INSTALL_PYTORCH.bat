@echo off
echo ============================================================
echo Clean PyTorch Reinstall for CUDA 12.1
echo ============================================================
echo.
echo This will DELETE old PyTorch and reinstall fresh
echo.
pause

cd /d "%~dp0"

REM Force D drive
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TORCH_HOME=%~dp0cache

echo.
echo Removing old PyTorch folders completely...
rmdir /s /q python_packages\torch
rmdir /s /q python_packages\torchvision
rmdir /s /q python_packages\torchaudio
rmdir /s /q python_packages\torch-*.dist-info
rmdir /s /q python_packages\torchvision-*.dist-info
rmdir /s /q python_packages\torchaudio-*.dist-info
rmdir /s /q python_packages\functorch

echo.
echo Installing fresh PyTorch CUDA 12.1...
python -m pip install --cache-dir pip_cache --target python_packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Testing...
python -c "import sys; sys.path.insert(0, 'python_packages'); import torch; print(''); print('='*60); print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('CUDA Version:', torch.version.cuda); print('='*60); print('')"

echo.
pause
