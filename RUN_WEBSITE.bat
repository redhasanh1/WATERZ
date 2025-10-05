@echo off
echo =====================================
echo AI Watermark Remover - Web Interface
echo =====================================
echo.

cd /d "%~dp0web"

echo Starting Flask server...
echo Server will be available at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
