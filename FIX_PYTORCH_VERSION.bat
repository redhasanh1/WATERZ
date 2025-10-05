@echo off
echo ============================================================
echo Fixing PyTorch Version Mismatch
echo ============================================================
echo.
echo Current issue: torch==2.7 incompatible with torchvision==0.20
echo.
echo Solution: Install matching versions for CUDA 12.1 (GTX 1660 Ti)
echo.
pause

cd /d "%~dp0"

REM Force D drive
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TORCH_HOME=%~dp0cache

echo.
echo Uninstalling old PyTorch versions...
python -m pip uninstall -y torch torchvision torchaudio

echo.
echo Installing compatible PyTorch 2.5.1 + torchvision 0.20.1 for CUDA 12.1...
python -m pip install --cache-dir pip_cache --target python_packages torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

echo.
echo Testing installation...
python -c "import sys; sys.path.insert(0, 'python_packages'); import torch; import torchvision; print(''); print('='*60); print('✅ PyTorch:', torch.__version__); print('✅ torchvision:', torchvision.__version__); print('✅ CUDA Available:', torch.cuda.is_available()); print('✅ GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('='*60); print('')"

echo.
echo If you see matching versions above, you're good!
echo.
pause
