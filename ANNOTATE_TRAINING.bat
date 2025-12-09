@echo off
echo ========================================
echo    YOLO Training Annotation Tool
echo ========================================
echo.
echo Draw boxes around watermarks in videos
echo Progress saves after each video!
echo.
echo Controls:
echo   - Drag mouse: Draw box
echo   - Right-click: Undo last box
echo   - 'n': Next video (save + continue)
echo   - 's': Skip video
echo   - 'r': Reset boxes
echo   - 'q': Quit (progress saved)
echo.
echo ========================================
echo.

cd /d D:\watermarkz
python annotate_training_videos.py

echo.
pause
