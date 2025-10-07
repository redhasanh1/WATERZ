@echo off
echo ================================================================
echo FIXING WINDOWS FIREWALL FOR LOCALTUNNEL
echo ================================================================
echo.
echo This will allow Node.js to connect through your firewall
echo.
pause

echo Adding inbound rule...
netsh advfirewall firewall add rule name="Node.js Localtunnel In" dir=in action=allow program="%~dp0node\node.exe" enable=yes

echo.
echo Adding outbound rule...
netsh advfirewall firewall add rule name="Node.js Localtunnel Out" dir=out action=allow program="%~dp0node\node.exe" enable=yes

echo.
echo ================================================================
echo DONE!
echo ================================================================
echo.
echo Firewall rules added for Node.js
echo.
echo Now run START_ALL.bat to test localtunnel
echo.
pause
