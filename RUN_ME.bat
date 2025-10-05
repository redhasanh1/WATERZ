@echo off
echo ========================================
echo Sora Watermark Removal Tool
echo ========================================
echo.

cd /d "%~dp0"

REM Force ALL temp/cache to D drive
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TMPDIR=%~dp0temp
set TORCH_HOME=%~dp0pip_cache
set XDG_CACHE_HOME=%~dp0pip_cache
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir temp 2>nul

python remove_watermark.py

pause
