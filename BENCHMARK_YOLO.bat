@echo off
REM Simple YOLO Benchmark: .pt vs .engine speedup test

echo ============================================================
echo YOLO BENCHMARK - PyTorch vs TensorRT
echo ============================================================
echo.

REM Check if video path provided
if "%~1"=="" (
    echo Usage: BENCHMARK_YOLO.bat ^<path_to_video^>
    echo.
    echo Examples:
    echo   BENCHMARK_YOLO.bat uploads\test_video.mp4
    echo   BENCHMARK_YOLO.bat temp\video_cache\05d90319-fef7-41fc-8103-449ccae86845.mp4
    echo.
    pause
    exit /b 1
)

REM Run benchmark
"C:\Users\Hasan\AppData\Local\Programs\Python\Python311\python.exe" benchmark_yolo.py "%~1"

echo.
echo ============================================================
echo Benchmark complete!
echo ============================================================
pause
