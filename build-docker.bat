@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo ==========================================
echo   Building Docker Image for Watermark Removal
echo ==========================================
echo.

REM Configuration
set IMAGE_NAME=humblewoslayer/watermarkz
set TAG=latest

echo 📦 Building image: %IMAGE_NAME%:%TAG%
echo.

REM Build the Docker image
docker build -t %IMAGE_NAME%:%TAG% .

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Build failed!
    pause
    exit /b 1
)

echo.
echo ✅ Build complete!
echo.
echo ==========================================
echo   Push to Docker Hub?
echo ==========================================
echo.
echo Image: %IMAGE_NAME%:%TAG%
echo.
set /p PUSH="Push to Docker Hub? (y/n): "

if /i "%PUSH%"=="y" (
    echo.
    echo 🚀 Pushing to Docker Hub...
    docker push %IMAGE_NAME%:%TAG%

    if %ERRORLEVEL% NEQ 0 (
        echo.
        echo ❌ Push failed!
        echo    Make sure you're logged in: docker login
        pause
        exit /b 1
    )

    echo.
    echo ✅ Push complete!
    echo.
    echo ==========================================
    echo   Image available at:
    echo   docker pull %IMAGE_NAME%:%TAG%
    echo ==========================================
) else (
    echo.
    echo ℹ️  Skipping push to Docker Hub
    echo.
    echo To push later, run:
    echo   docker push %IMAGE_NAME%:%TAG%
)

echo.
echo ==========================================
echo   Test Locally (optional)
echo ==========================================
echo.
echo To test the image locally:
echo.
echo docker run --gpus all -e REDIS_URL="redis://..." %IMAGE_NAME%:%TAG%
echo.

pause
