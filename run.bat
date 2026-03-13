@echo off
:: ============================================================
:: run.bat – One-click startup script for Windows
:: Double-click this file OR run it from the terminal:
::   .\run.bat
:: ============================================================

title Semantic Similarity App

echo ======================================
echo   Semantic Similarity Measurement App
echo ======================================
echo.

:: Activate the virtual environment (created in .venv folder)
echo [1/3] Activating virtual environment...
call .venv\Scripts\activate.bat

:: Install / update dependencies (skips if already installed)
echo [2/3] Checking dependencies...
pip install -r requirements.txt --quiet

echo.
echo [3/3] Starting server at http://localhost:8000
echo       Press CTRL+C to stop.
echo.

:: Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
