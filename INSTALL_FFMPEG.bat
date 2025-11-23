@echo off
echo ========================================
echo Installing FFmpeg (static-ffmpeg)
echo ========================================
echo.
echo This installs a portable FFmpeg that doesn't require system installation
echo.

cd /d "%~dp0"

echo Installing static-ffmpeg via pip...
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -m pip install static-ffmpeg

echo.
echo ========================================
echo Testing FFmpeg...
echo ========================================
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -c "from static_ffmpeg import run; ffmpeg, ffprobe = run.get_or_fetch_platform_executables_else_raise(); print('FFmpeg:', ffmpeg); print('FFprobe:', ffprobe)"

echo.
echo ========================================
echo FFmpeg installed successfully!
echo ========================================
pause
