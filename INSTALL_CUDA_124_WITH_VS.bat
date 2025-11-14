@echo off
REM Download and install CUDA 12.4.1 with Visual Studio integration
setlocal

set "CUDA_URL=https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_551.78_windows.exe"
set "CUDA_EXE=%~dp0cuda_12.4.1_windows.exe"

echo ----- CUDA 12.4.1 Offline Installer -----
echo Source: %CUDA_URL%
echo Target: %CUDA_EXE%

REM Download if missing (avoid parentheses blocks to prevent parse issues on some shells)
IF EXIST "%CUDA_EXE%" GOTO SKIP_DL
echo Downloading installer (this may take several minutes)...
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri '%CUDA_URL%' -OutFile '%CUDA_EXE%'" || GOTO DL_FAIL
IF NOT EXIST "%CUDA_EXE%" GOTO DL_FAIL

:SKIP_DL
echo Installing CUDA 12.4.1 with Visual Studio integration...
"%CUDA_EXE%" -s nvcc_12.4 nvml_devtools visual_studio_integration || GOTO INST_FAIL

echo CUDA 12.4.1 installation completed.
echo Verify with the following commands (run in a new terminal):
echo   where nvcc
echo   echo %%CUDA_PATH%%
echo   dir "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\extras\visual_studio_integration\MSBuildExtensions"
echo   dir "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\*\BuildCustomizations\cuda*"
GOTO END

:DL_FAIL
echo Failed to download CUDA installer. Aborting.
exit /b 1

:INST_FAIL
echo CUDA installer returned error code %ERRORLEVEL%.
exit /b %ERRORLEVEL%

:END
endlocal
