@echo off
setlocal enabledelayedexpansion

echo ================================
echo Video to 10 FPS Converter
echo ================================
echo.

REM Create a temporary VBS script for file picker
set "vbs_script=%temp%\file_picker.vbs"

echo Set objDialog = CreateObject("SAFRDialogs.FileSave") > "%vbs_script%"
echo objDialog.FileName = "Select Video File" >> "%vbs_script%"
echo objDialog.FileType = "All Video Files" >> "%vbs_script%"
echo objDialog.Filter = "Video Files|*.mp4;*.avi;*.mov;*.mkv;*.flv;*.wmv;*.webm;*.m4v;*.mpeg;*.mpg;*.3gp;*.ts;*.mts;*.m2ts|All Files|*.*" >> "%vbs_script%"
echo If objDialog.Show Then >> "%vbs_script%"
echo     WScript.Echo objDialog.FileName >> "%vbs_script%"
echo End If >> "%vbs_script%"

REM Better approach - use PowerShell for file picker
set "ps_script=%temp%\file_picker.ps1"

echo Add-Type -AssemblyName System.Windows.Forms > "%ps_script%"
echo $FileBrowser = New-Object System.Windows.Forms.OpenFileDialog -Property @{ >> "%ps_script%"
echo     Title = 'Select Video File to Convert' >> "%ps_script%"
echo     Filter = 'Video Files (*.mp4;*.avi;*.mov;*.mkv;*.flv;*.wmv;*.webm;*.m4v;*.mpeg;*.mpg;*.3gp;*.ts;*.mts;*.m2ts)^|*.mp4;*.avi;*.mov;*.mkv;*.flv;*.wmv;*.webm;*.m4v;*.mpeg;*.mpg;*.3gp;*.ts;*.mts;*.m2ts^|All Files (*.*)^|*.*' >> "%ps_script%"
echo } >> "%ps_script%"
echo $result = $FileBrowser.ShowDialog() >> "%ps_script%"
echo if ($result -eq 'OK') { >> "%ps_script%"
echo     Write-Output $FileBrowser.FileName >> "%ps_script%"
echo } >> "%ps_script%"

REM Run PowerShell script and capture the selected file
for /f "delims=" %%i in ('powershell -ExecutionPolicy Bypass -File "%ps_script%"') do set "input_video=%%i"

REM Clean up temporary scripts
del "%ps_script%" 2>nul
del "%vbs_script%" 2>nul

REM Check if a file was selected
if not defined input_video (
    echo No file selected. Exiting...
    pause
    exit /b 1
)

if not exist "%input_video%" (
    echo File not found: %input_video%
    pause
    exit /b 1
)

echo Selected video: %input_video%
echo.

REM Get the directory and filename
for %%F in ("%input_video%") do (
    set "video_dir=%%~dpF"
    set "video_name=%%~nF"
    set "video_ext=%%~xF"
)

REM Create output filename
set "output_video=%video_dir%%video_name%_10fps.mp4"

echo Converting to 10 FPS (keeping original resolution)...
echo.

REM Use Python script with static-ffmpeg (same approach as 16fps converter)
"%LOCALAPPDATA%\Programs\Python\Python312\python.exe" "%~dp0convert_to_10fps.py" "%input_video%"

if %errorlevel% equ 0 (
    echo.
    echo ================================
    echo Conversion complete!
    echo ================================
    echo Output: %output_video%
    echo.
) else (
    echo.
    echo ================================
    echo Error during conversion!
    echo ================================
    echo.
)

pause
