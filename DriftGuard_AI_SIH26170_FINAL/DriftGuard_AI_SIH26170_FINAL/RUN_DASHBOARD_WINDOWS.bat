@echo off
setlocal
cd /d "%~dp0"
echo =============================================
echo DriftGuard AI - SIH26170 Prototype
echo =============================================

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.11+ and enable "Add Python to PATH".
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/4] Creating local virtual environment...
  python -m venv .venv
)

echo [2/4] Installing/updating required packages...
".venv\Scripts\python.exe" -m pip install --upgrade -r requirements.txt
if errorlevel 1 (
  echo Package installation failed. Check your internet connection and try again.
  pause
  exit /b 1
)

echo [3/4] Checking ML model compatibility...
".venv\Scripts\python.exe" src\model_preflight.py
if errorlevel 1 (
  echo Model compatibility check failed.
  echo Try running REBUILD_MODELS_WINDOWS.bat once.
  pause
  exit /b 1
)

echo [4/4] Starting DriftGuard AI...
".venv\Scripts\python.exe" -m streamlit run app.py
endlocal
