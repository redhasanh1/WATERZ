@echo off
setlocal ENABLEEXTENSIONS

REM Launch two Celery workers in separate consoles (w1-w2).
cd /d "%~dp0"

echo ===========================================================
echo  Launching 2 Celery workers (w1 - w2)
echo  Project root: %CD%
echo  Each worker inherits TensorRT/Python path fixes.
echo ===========================================================
echo(

for %%I in (1 2) do (
    start "celery-w%%I" "%~dp0run_celery_worker.bat" w%%I 0
    REM Give Windows a beat so console titles stay ordered
    timeout /t 1 >nul
)

echo All Celery workers launched. Close this window if desired.

endlocal
