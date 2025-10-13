@echo off
setlocal enabledelayedexpansion

echo ============================================
echo Reduce Resolution, FPS, and Compression
echo - Random video from videostotrain
echo - ENTIRE LENGTH at 480p height, 15 FPS
echo - Preserves aspect ratio, x264 fast preset, CRF 28, copy audio
echo ============================================
echo.

set "script_dir=%~dp0"
set "folder=%script_dir%videostotrain"
set "outdir=%script_dir%optimized"
if not exist "%outdir%" mkdir "%outdir%" >nul 2>&1

set /a count=0

REM Collect videos (common extensions)
for %%x in (mp4 mov mkv avi) do (
  for %%f in ("%folder%\*.%%x") do (
    if exist "%%~ff" (
      set /a count+=1
      set "video!count!=%%~ff"
    )
  )
)

if %count%==0 (
  echo ERROR: No videos found in %folder%
  pause
  exit /b 1
)

REM Pick random video
set /a "rand=%RANDOM% %% count + 1"
set "input_video=!video%rand%!"

echo Found %count% videos
echo Selected: !input_video!
echo.

REM Derive output file name
for %%F in ("!input_video!") do set "NAME=%%~nF"
set "outfile=%outdir%\%NAME%_480p15.mp4"

echo === Reducing resolution, fps, and compression ===
echo 1) Lower resolution to 480p height (keeps aspect ratio)
echo 2) Cut FPS to 15
echo 3) Use fast preset
echo.

REM Keep audio stream as-is for speed; use faststart for web playback
ffmpeg -hide_banner -loglevel error -y ^
  -i "!input_video!" ^
  -vf "scale=-2:480,fps=15" ^
  -c:v libx264 -preset fast -crf 28 -pix_fmt yuv420p -movflags +faststart ^
  -c:a copy ^
  "!outfile!"

if %ERRORLEVEL%==0 (
  echo.
  echo ✓ Done: "!outfile!"
) else (
  echo.
  echo ERROR: ffmpeg failed
)

echo.
pause
