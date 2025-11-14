@echo off
REM Benchmark RAFT: PyTorch RAFT_bi vs TensorRT FastFlowNet engine
setlocal

call "%~dp0SETUP_TRT_ENV.bat"

echo RAFT BENCHMARK - PyTorch vs TensorRT
echo.

if "%~1"=="" (
  echo Usage: BENCHMARK_RAFT.bat ^<path_to_video^>
  echo Example:
  echo   BENCHMARK_RAFT.bat uploads\test_video.mp4
  goto :eof
)

set VIDEO=%~1

REM Adjust Python path if needed (keeping consistent with other scripts)
set PY311="C:\Users\Hasan\AppData\Local\Programs\Python\Python311\python.exe"

%PY311% benchmark_raft.py "%VIDEO%"

if errorlevel 1 (
  echo Benchmark failed.
  exit /b 1
)

echo.
echo Benchmark complete!
endlocal

