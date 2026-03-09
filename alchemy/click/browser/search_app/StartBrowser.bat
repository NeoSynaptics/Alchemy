@echo off
title AlchemyBrowser Server
cd /d "%~dp0"

echo.
echo  Installing / checking dependencies...
pip install -q -r requirements_browser.txt

echo.
echo  Starting AlchemyBrowser at http://localhost:8055
echo  Press Ctrl+C to stop.
echo.

python browser_server.py
pause
