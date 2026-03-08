@echo off
title Alchemy Environment
echo ========================================
echo   Alchemy — Starting Environment
echo ========================================
echo.

:: Start backend (uvicorn) in background
echo [1/2] Starting Alchemy backend on :8000...
start "Alchemy Backend" cmd /k "cd /d C:\Users\info\GitHub\Alchemy && .venv\Scripts\activate && uvicorn alchemy.server:app --host 0.0.0.0 --port 8000 --reload"

:: Give backend a moment to boot
timeout /t 3 /nobreak >nul

:: Start UI dev server in background
echo [2/2] Starting Alchemy UI on :5173...
start "Alchemy UI" cmd /k "cd /d C:\Users\info\GitHub\Alchemy\ui && npm run dev"

:: Wait then open browser
timeout /t 4 /nobreak >nul
echo.
echo Opening Alchemy UI in browser...
start http://localhost:5173

echo.
echo Alchemy is running.
echo   Backend: http://localhost:8000
echo   UI:      http://localhost:5173
echo.
echo Close this window — servers run independently.
timeout /t 5
