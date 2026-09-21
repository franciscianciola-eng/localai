@echo off
REM One-step launcher for Windows. Downloads the models the first time
REM (needs internet), then serves the app on http://localhost:8000.
cd /d "%~dp0"

where python >nul 2>nul && (set PY=python) || (where py >nul 2>nul && (set PY=py))
if "%PY%"=="" (
  echo Python 3 is required but was not found. Install it from https://www.python.org/downloads/ and re-run.
  pause
  exit /b 1
)

if not exist "models\mlc-ai" (
  echo No models found yet - downloading them once ^(needs internet^)...
  %PY% download-models.py || (echo Download failed. & pause & exit /b 1)
)

%PY% serve.py
pause
