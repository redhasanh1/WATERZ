param(
    [string]$InputVideo = "C:\SurFastVideos\comedy.mp4",
    [string]$OutputVideo = "C:\SurFastVideos\comedy_20min.mp4",
    [string]$StartTime = "00:00:00"
)

Write-Host "[CUT] Cutting 20 minutes from: $InputVideo" -ForegroundColor Cyan
Write-Host "[CUT] Start time: $StartTime" -ForegroundColor Cyan
Write-Host "[CUT] Output: $OutputVideo" -ForegroundColor Cyan

# Get FFmpeg from static-ffmpeg
$pythonPath = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$ffmpegCmd = & $pythonPath -c @"
from static_ffmpeg import run
ffmpeg_path, _ = run.get_or_fetch_platform_executables_else_raise()
print(ffmpeg_path)
"@

Write-Host "[CUT] Using FFmpeg: $ffmpegCmd" -ForegroundColor Gray

# Cut 20 minutes (1200 seconds) starting from $StartTime with GPU acceleration
& $ffmpegCmd -hwaccel cuda -ss "$StartTime" -i "$InputVideo" -t 1200 -c:v h264_nvenc -preset fast -b:v 5M -c:a copy "$OutputVideo"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[OK] Done! First 20 minutes saved to: $OutputVideo" -ForegroundColor Green
    Write-Host "[OK] Now you can test with: .\RUN_SAM2_WSL2_GUI.ps1 `"$OutputVideo`"" -ForegroundColor Green
} else {
    Write-Host "`n[ERROR] FFmpeg failed with exit code: $LASTEXITCODE" -ForegroundColor Red
}
