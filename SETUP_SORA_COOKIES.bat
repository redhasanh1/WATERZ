@echo off
echo ============================================================
echo Setup Sora Authentication Cookies
echo ============================================================
echo.
echo This will open a browser window for you to login to ChatGPT Sora.
echo After logging in, return to this window and press ENTER.
echo.
pause

cd downz
python save_cookies.py

echo.
echo ============================================================
echo Setup Complete!
echo ============================================================
echo.
echo Your cookies have been saved. The watermark removal site
echo can now download videos from ChatGPT Sora.
echo.
pause
