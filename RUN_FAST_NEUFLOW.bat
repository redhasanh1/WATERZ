@echo off
cd /d "%~dp0"

echo ============================================================
echo FAST NEUFLOW PROCESSING (10-70x faster than RAFT)
echo ============================================================
echo.
echo This will:
echo   1. Let you pick a video file
echo   2. Use existing masks from temp_sam2_masks/
echo   3. Process with ProPainter using NeuFlow TensorRT (FASTEST!)
echo   4. Save result to results/
echo.
echo ============================================================
echo.

"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" FAST_NEUFLOW_PROCESS.py

pause
