@echo off
echo ========================================================
echo YOLO Manual Labeling Tool
echo ========================================================
echo.
echo Draw boxes around watermarks in video frames
echo This will create a training dataset for YOLOv8
echo.

python label_tool.py

echo.
pause
