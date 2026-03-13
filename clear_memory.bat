@echo off
echo ========================================
echo   MEMORY CLEANUP - VRAM + RAM + WSL
echo ========================================

echo.
echo [1/5] Killing Python processes hogging GPU...
taskkill /F /IM python.exe 2>nul
taskkill /F /IM python3.exe 2>nul
taskkill /F /IM pythonw.exe 2>nul

echo.
echo [2/5] Killing CUDA/GPU processes...
taskkill /F /IM nvidia-smi.exe 2>nul
taskkill /F /IM nvcontainer.exe 2>nul

echo.
echo [3/5] Skipping WSL shutdown (Docker uses WSL2)...
echo Docker containers preserved

echo.
echo [4/5] Flushing Windows RAM...
echo Skipped (nothing safe to clear without killing apps)

echo.
echo [5/5] Resetting CUDA context...
:: This forces CUDA to release memory
nvidia-smi --gpu-reset 2>nul || echo GPU reset not available (need admin)

echo.
echo ========================================
echo   CLEANUP COMPLETE
echo ========================================
echo.
echo Checking GPU memory now:
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv
echo.
pause
