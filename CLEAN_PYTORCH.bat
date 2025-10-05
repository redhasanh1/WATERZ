@echo off
echo ============================================================
echo Cleaning Old PyTorch Versions
echo ============================================================
echo.

cd /d "%~dp0"

echo Removing old PyTorch dist-info folders...
rmdir /s /q python_packages\torch-2.7.1+cu118.dist-info 2>nul
rmdir /s /q python_packages\torch-2.8.0.dist-info 2>nul
rmdir /s /q python_packages\torchvision-0.22.1+cu118.dist-info 2>nul
rmdir /s /q python_packages\torchaudio-2.7.1+cu118.dist-info 2>nul

echo.
echo Testing new PyTorch version...
python -c "import sys; sys.path.insert(0, 'python_packages'); import torch; print(''); print('='*60); print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('CUDA Version:', torch.version.cuda); print('='*60); print('')"

echo.
pause
