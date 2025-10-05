@echo off
cd /d "%~dp0"

REM Force ALL to D drive
set PYTHONPATH=%~dp0python_packages
set PIP_CACHE_DIR=%~dp0pip_cache
set TEMP=%~dp0temp
set TMP=%~dp0temp
set TORCH_HOME=%~dp0pip_cache

echo Fixing LaMa dependencies...
echo.

echo Uninstalling incompatible huggingface_hub...
python -m pip uninstall -y huggingface_hub --target python_packages 2>nul

echo Installing compatible huggingface_hub version 0.16.4...
python -m pip install --target python_packages --cache-dir pip_cache --no-warn-script-location huggingface_hub==0.16.4

echo.
echo Done! Now test with: python remove_with_lama.py
pause
