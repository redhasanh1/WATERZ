@echo off
echo ============================================================
echo Installing PyTorch for CUDA 12.6 (Your GTX 1660 Ti)
echo ============================================================
echo.
echo Detected: CUDA 12.6
echo GPU: GTX 1660 Ti (6GB)
echo.
pause

cd /d "%~dp0"

REM Force D drive
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TMPDIR=%~dp0temp
set TORCH_HOME=%~dp0cache
set XDG_CACHE_HOME=%~dp0cache

mkdir temp 2>nul
mkdir cache 2>nul
mkdir pip_cache 2>nul
mkdir python_packages 2>nul

echo.
echo Installing PyTorch with CUDA 12.1 support (compatible with 12.6)...
echo This will download to: D:\github\RoomFinderAI\watermarkz\pip_cache
echo.

python -m pip install --cache-dir pip_cache --target python_packages torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Testing GPU detection...
python -c "import sys; sys.path.insert(0, 'python_packages'); import torch; print(''); print('='*60); print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); print('VRAM:', f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if torch.cuda.is_available() else 'N/A'); print('CUDA Version (PyTorch):', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('='*60); print('')"

echo.
echo If you see "GTX 1660 Ti" and "True" above, you're ready!
echo.
pause
