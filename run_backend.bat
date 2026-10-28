@echo off
title DefectVision FastAPI Backend (Port 8000)
cd /d "%~dp0backend"
echo ======================================================================
echo          Starting DefectVision FastAPI Backend Server
echo          URL: http://127.0.0.1:8000  (API Docs: /docs)
echo ======================================================================
py -3.10 -m uvicorn main:app --reload --port 8000
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Backend crashed or exited with error code %ERRORLEVEL%.
    pause
)
