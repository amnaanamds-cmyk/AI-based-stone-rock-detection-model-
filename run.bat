@echo off
REM RockMap one-command launcher (Windows)
REM   run.bat            install (first time), build the demo and open the dashboard
REM   run.bat --real     also download a real 40 x 40 km Gilgit region (needs internet)
REM   run.bat serve      just start the dashboard on existing data
setlocal
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
if not exist .venv (
  echo ^>^> Creating virtual environment .venv
  %PY% -m venv .venv || (echo Python 3.9+ is required: https://www.python.org/downloads/ & exit /b 1)
)
call .venv\Scripts\activate.bat
python -c "import rockmap, torch" >nul 2>nul
if errorlevel 1 (
  echo ^>^> Installing RockMap and dependencies - first run only, a few minutes
  python -m pip install --upgrade pip
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  python -m pip install -e ".[dev]"
)
if /I "%1"=="serve" (
  rockmap serve --open
) else (
  rockmap quickstart %*
)
