@echo off
setlocal
cd /d "%~dp0"
echo =============================================
echo DriftGuard - Rebuild ML Models Locally
echo =============================================

if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Run RUN_DASHBOARD_WINDOWS.bat first.
  pause
  exit /b 1
)

echo Updating project dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade -r requirements.txt
if errorlevel 1 (
  echo Dependency update failed.
  pause
  exit /b 1
)

echo Rebuilding all models with this computer's installed scikit-learn...
".venv\Scripts\python.exe" src\train_models.py --data data\demo_burn_in.csv --models models
if errorlevel 1 (
  echo Model rebuild failed.
  pause
  exit /b 1
)

echo.
echo Model rebuild complete. You can now start DriftGuard.
pause
endlocal
