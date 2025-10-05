@echo off 
cd /d "%~dp0" 
 
REM Force D drive temp/cache 
set TEMP=%~dp0temp 
set TMP=%~dp0temp 
set TORCH_HOME=%~dp0cache 
 
echo Starting Celery worker (optimized for GTX 1660 Ti)... 
echo Using concurrency=1 (leaves GPU available for gaming) 
celery -A server_production.celery worker --loglevel=info --concurrency=1 
