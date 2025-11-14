@echo off 
cd /d "%~dp0" 
echo Starting Redis on D drive... 
redis-server.exe redis.conf 
