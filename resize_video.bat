@echo off
echo Resizing video for 6GB GPU...
ffmpeg -i sora_with_watermark.mp4 -vf scale=352:640 sora_small.mp4
echo Done! Now run RUN_ME.bat
pause
