@echo off
setlocal ENABLEEXTENSIONS

REM Launch three Celery workers in separate consoles (w1-w3).
cd /d "%~dp0"

echo ===========================================================
echo  Launching 3 Celery workers (w1 - w3)
echo  Project root: %CD%
echo  Each worker inherits TensorRT/Python path fixes.
echo ===========================================================
echo(

for %%I in (1 2 3) do (
    start "celery-w%%I" "%~dp0run_celery_worker.bat" w%%I 0
    REM Give Windows a beat so console titles stay ordered
    timeout /t 1 >nul
)

echo All Celery workers launched. Close this window if desired.

endlocal
