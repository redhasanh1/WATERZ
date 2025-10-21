@echo off
setlocal ENABLEEXTENSIONS

REM Optional first argument = worker name (defaults to w<random>)
set "WORKER_NAME=%~1"
if "%WORKER_NAME%"=="" set "WORKER_NAME=w%RANDOM%"

REM Resolve repo root (this script lives inside watermarkz folder)
cd /d "%~dp0"
set "ROOT_DIR=%CD%"

REM Pin temp/cache directories to the project folder
set "PIP_CACHE_DIR=%ROOT_DIR%\pip_cache"
set "TEMP=%ROOT_DIR%\temp"
set "TMP=%ROOT_DIR%\temp"
set "TMPDIR=%ROOT_DIR%\temp"
set "TORCH_HOME=%ROOT_DIR%\cache"
set "XDG_CACHE_HOME=%ROOT_DIR%\cache"
set "OPENCV_TEMP_PATH=%ROOT_DIR%\temp"

REM Share project + bundled packages on Python path
set "PYTHONPATH=%ROOT_DIR%;%ROOT_DIR%\python_packages"

REM Append TensorRT/Torch libs to PATH if present
if exist "%ROOT_DIR%\TensorRT-10.7.0.23\lib" (
    set "PATH=%ROOT_DIR%\TensorRT-10.7.0.23\lib;%PATH%"
)
if exist "%ROOT_DIR%\python_packages" (
    set "PATH=%ROOT_DIR%\python_packages;%PATH%"
)
if exist "%ROOT_DIR%\python_packages\torch\lib" (
    set "PATH=%ROOT_DIR%\python_packages\torch\lib;%PATH%"
)

REM Disable buffering so logs stream live into the console
set "PYTHONUNBUFFERED=1"
set "PYTHONIOENCODING=utf-8"

echo ===========================================================
echo  Starting Celery worker %WORKER_NAME%
echo  Root: %ROOT_DIR%
if exist "%ROOT_DIR%\TensorRT-10.7.0.23\lib" (
  echo  TensorRT libs: %ROOT_DIR%\TensorRT-10.7.0.23\lib
) else (
  echo  TensorRT libs: NOT FOUND (will fall back to PyTorch)
)
echo ===========================================================
echo(

python -u -m celery -A server_production:celery worker -P solo -n %WORKER_NAME% -l INFO

endlocal
