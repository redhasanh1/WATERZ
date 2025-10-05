@echo off
echo ============================================================
echo VERIFICATION - Checking if C drive is safe
echo ============================================================
echo.

cd /d "%~dp0"

echo Checking common C drive temp locations...
echo.

REM Check C drive temp folders
set "CTMP=C:\Users\%USERNAME%\AppData\Local\Temp"
set "CPIP=C:\Users\%USERNAME%\AppData\Local\pip"
set "CCACHE=C:\Users\%USERNAME%\.cache"
set "CTORCH=C:\Users\%USERNAME%\.torch"

echo [1] C:\Users\%USERNAME%\AppData\Local\Temp
dir "%CTMP%\*watermark*" 2>nul && echo    ⚠️  WARNING: Watermark files found! || echo    ✅ Clean
dir "%CTMP%\*yolo*" 2>nul && echo    ⚠️  WARNING: YOLO files found! || echo    ✅ Clean
dir "%CTMP%\*torch*" 2>nul && echo    ⚠️  WARNING: Torch files found! || echo    ✅ Clean
echo.

echo [2] C:\Users\%USERNAME%\AppData\Local\pip
if exist "%CPIP%" (
    dir "%CPIP%" | find "File(s)" && echo    ⚠️  WARNING: Pip cache on C drive! || echo    ✅ Clean
) else (
    echo    ✅ Folder doesn't exist (good!)
)
echo.

echo [3] C:\Users\%USERNAME%\.cache
if exist "%CCACHE%" (
    dir "%CCACHE%\torch" 2>nul && echo    ⚠️  WARNING: Torch cache on C drive! || echo    ✅ Clean
    dir "%CCACHE%\huggingface" 2>nul && echo    ⚠️  WARNING: HuggingFace cache on C drive! || echo    ✅ Clean
) else (
    echo    ✅ Folder doesn't exist (good!)
)
echo.

echo [4] C:\Users\%USERNAME%\.torch
if exist "%CTORCH%" (
    echo    ⚠️  WARNING: PyTorch home on C drive!
    dir "%CTORCH%"
) else (
    echo    ✅ Folder doesn't exist (good!)
)
echo.

echo ============================================================
echo Checking D drive (watermarkz folder)...
echo ============================================================
echo.

echo Current directory: %CD%
echo.

echo [✓] Folders that SHOULD exist on D drive:
if exist temp (echo    ✅ temp) else (echo    ❌ temp - RUN INSTALL_ALL_LOCAL.bat first!)
if exist cache (echo    ✅ cache) else (echo    ❌ cache - RUN INSTALL_ALL_LOCAL.bat first!)
if exist pip_cache (echo    ✅ pip_cache) else (echo    ❌ pip_cache - RUN INSTALL_ALL_LOCAL.bat first!)
if exist uploads (echo    ✅ uploads) else (echo    ❌ uploads - RUN INSTALL_ALL_LOCAL.bat first!)
if exist results (echo    ✅ results) else (echo    ❌ results - RUN INSTALL_ALL_LOCAL.bat first!)
if exist python_packages (echo    ✅ python_packages) else (echo    ❌ python_packages - RUN INSTALL_ALL_LOCAL.bat first!)
if exist redis_data (echo    ✅ redis_data) else (echo    ❌ redis_data - RUN INSTALL_ALL_LOCAL.bat first!)
echo.

echo [✓] Checking folder sizes on D drive:
echo.
for /d %%d in (temp cache pip_cache uploads results python_packages redis_data) do (
    if exist %%d (
        echo    %%d:
        dir %%d | find "File(s)" | find "bytes"
    )
)
echo.

echo ============================================================
echo Environment Variables Check
echo ============================================================
echo.

echo Current session environment:
echo    TEMP=%TEMP%
echo    TMP=%TMP%
echo    PIP_CACHE_DIR=%PIP_CACHE_DIR%
echo    TORCH_HOME=%TORCH_HOME%
echo.

if "%TEMP%"=="%~dp0temp" (
    echo ✅ TEMP points to D drive
) else (
    echo ⚠️  WARNING: TEMP still points to C drive!
    echo    Run: set TEMP=%~dp0temp
)

if "%TORCH_HOME%"=="%~dp0cache" (
    echo ✅ TORCH_HOME points to D drive
) else (
    echo ⚠️  WARNING: TORCH_HOME not set!
    echo    Run: set TORCH_HOME=%~dp0cache
)
echo.

echo ============================================================
echo FINAL VERDICT
echo ============================================================
echo.

if exist "%CCACHE%\torch" (
    echo ❌ FAILED - Files found on C drive!
    echo    Solution: Delete C:\Users\%USERNAME%\.cache\torch
    echo    Then run INSTALL_ALL_LOCAL.bat again
) else if exist "%CTORCH%" (
    echo ❌ FAILED - PyTorch home on C drive!
    echo    Solution: Delete C:\Users\%USERNAME%\.torch
    echo    Then run INSTALL_ALL_LOCAL.bat again
) else if not exist temp (
    echo ❌ FAILED - D drive folders not created!
    echo    Solution: Run INSTALL_ALL_LOCAL.bat first
) else (
    echo ✅✅✅ SUCCESS - C DRIVE IS SAFE! ✅✅✅
    echo.
    echo Everything is on D drive:
    echo    D:\github\RoomFinderAI\watermarkz\
    echo.
    echo You can now run START_ALL.bat safely!
)

echo.
echo ============================================================
pause
