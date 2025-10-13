@echo off
setlocal enabledelayedexpansion

echo ============================================
echo Random Video Optimizer (Super Fast)
echo - Picks a random video from videostotrain
echo - Processes ENTIRE VIDEO, downsamples resolution and FPS
echo - Uses ultrafast encode, keeps audio (copy)
echo ============================================
echo.

set "folder=videostotrain"
set "outdir=optimized"
if not exist "%outdir%" mkdir "%outdir%" >nul 2>&1

set /a count=0

REM Collect videos across common extensions
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
set "outfile=%outdir%\%NAME%_superfast.mp4"

echo === Super-fast optimization ===
echo - Full length
echo - Video: scale to 360p height, ~8 FPS
echo - Encode: x264 ultrafast, CRF 32, copy audio
echo.

REM Notes for speed:
REM - Using ultrafast preset (minimal CPU)
REM - Dropping audio (-an)
REM - Lower FPS and resolution to reduce compute/IO
REM - yuv420p for broad compatibility

ffmpeg -hide_banner -loglevel error -y ^
  -i "!input_video!" ^
  -vf "scale=-2:360,fps=8" ^
  -c:v libx264 -preset ultrafast -crf 32 -pix_fmt yuv420p -movflags +faststart ^
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
