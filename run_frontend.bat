@echo off
title DefectVision React Frontend (Port 5173)
cd /d "%~dp0frontend"
echo ======================================================================
echo          Starting DefectVision React + Vite Frontend
echo          URL: http://localhost:5173
echo ======================================================================
npm run dev
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Frontend crashed or exited with error code %ERRORLEVEL%.
    pause
)
