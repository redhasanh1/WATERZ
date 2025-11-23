@echo off
echo ================================================================================
echo CLEAN AND REBUILD ALL WORKER CACHES
echo ================================================================================
echo.
echo Step 1: Killing any running cache builder processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *build_worker_caches*" 2>nul
timeout /t 2 >nul

echo Step 2: Cleaning corrupted cache directories...
rd /s /q "temp\.torch_cache_0" 2>nul
rd /s /q "temp\.torch_cache_1" 2>nul
rd /s /q "temp\.torch_cache_2" 2>nul
rd /s /q "temp\.torch_cache_3" 2>nul
rd /s /q "temp\.triton_cache_0" 2>nul
rd /s /q "temp\.triton_cache_1" 2>nul
rd /s /q "temp\.triton_cache_2" 2>nul
rd /s /q "temp\.triton_cache_3" 2>nul

echo Step 3: Recreating cache directories...
mkdir "temp\.torch_cache_0" 2>nul
mkdir "temp\.torch_cache_1" 2>nul
mkdir "temp\.torch_cache_2" 2>nul
mkdir "temp\.torch_cache_3" 2>nul
mkdir "temp\.triton_cache_0" 2>nul
mkdir "temp\.triton_cache_1" 2>nul
mkdir "temp\.triton_cache_2" 2>nul
mkdir "temp\.triton_cache_3" 2>nul

echo.
echo Cache directories cleaned!
echo.

echo Step 4: Building fresh caches for 4 workers...
echo This will take approximately 100 seconds...
echo.

set PYTHONIOENCODING=utf-8
set USE_TORCH_COMPILE_RAFT=1

"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" build_worker_caches.py --workers 4

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ================================================================================
    echo SUCCESS! All worker caches built successfully!
    echo ================================================================================
    echo.
) else (
    echo.
    echo ================================================================================
    echo FAILED with exit code %ERRORLEVEL%
    echo ================================================================================
    echo.
)

pause
