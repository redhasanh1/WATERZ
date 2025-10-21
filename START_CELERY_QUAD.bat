@echo off
setlocal ENABLEEXTENSIONS

REM Launch four Celery workers in separate consoles (w1-w4).
cd /d "%~dp0"

echo ===========================================================
echo  Launching 4 Celery workers (w1 - w4)
echo  Project root: %CD%
echo  Each worker inherits TensorRT/Python path fixes.
echo ===========================================================
echo(

for %%I in (1 2 3 4) do (
    start "celery-w%%I" "%~dp0run_celery_worker.bat" w%%I
    REM Give Windows a beat so console titles stay ordered
    timeout /t 1 >nul
)

echo All Celery workers launched. Close this window if desired.

endlocal
