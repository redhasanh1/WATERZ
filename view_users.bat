@echo off
cd /d "%~dp0"
echo Running view_users.py...
echo.
python view_users.py
if errorlevel 1 (
    echo.
    echo ERROR: Script failed!
)
echo.
pause
