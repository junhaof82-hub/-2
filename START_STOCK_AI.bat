@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Stock AI MAX Launcher

set "PY_CMD="

rem Prefer Python 3.12 from the Windows py launcher.
py -3.12 -c "import sys" >nul 2>nul
if not errorlevel 1 set "PY_CMD=py -3.12"

rem Otherwise use python if it is version 3.11 or 3.12.
if not defined PY_CMD (
  python -c "import sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] < (3,13) else 1)" >nul 2>nul
  if not errorlevel 1 set "PY_CMD=python"
)

rem If Python is missing, try to install Python 3.12 with winget.
if not defined PY_CMD (
  echo.
  echo [1/4] Python 3.11 or 3.12 was not found.
  echo Trying to install Python 3.12 automatically...
  where winget >nul 2>nul
  if errorlevel 1 goto :python_missing
  winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
  if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PY_CMD=%LocalAppData%\Programs\Python\Python312\python.exe"
  if exist "%ProgramFiles%\Python312\python.exe" set "PY_CMD=%ProgramFiles%\Python312\python.exe"
  if not defined PY_CMD (
    py -3.12 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PY_CMD=py -3.12"
  )
)

if not defined PY_CMD goto :python_missing

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo [2/4] Creating the Stock AI environment. This only happens once...
  %PY_CMD% -m venv .venv
  if errorlevel 1 goto :fail
)

set "VENV_PY=.venv\Scripts\python.exe"

if not exist ".venv\.stock_ai_ready" (
  echo.
  echo [3/4] Installing Stock AI packages. This may take several minutes...
  "%VENV_PY%" -m pip install --upgrade pip
  if errorlevel 1 goto :fail
  "%VENV_PY%" -m pip install -r requirements.txt
  if errorlevel 1 goto :fail
  type nul > ".venv\.stock_ai_ready"
)

if not exist ".env" copy /Y ".env.example" ".env" >nul

echo.
echo [4/4] Starting Stock AI MAX...
echo Your browser should open automatically.
echo If it does not, open: http://localhost:8501
echo.
start "" cmd /c "timeout /t 3 /nobreak >nul & start http://localhost:8501"
"%VENV_PY%" -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
goto :end

:python_missing
echo.
echo Python could not be installed automatically.
echo A Python download page will open. Install Python 3.12, then run this file again.
start "" "https://www.python.org/downloads/"
pause
goto :end

:fail
echo.
echo ============================================================
echo Stock AI setup or startup did not finish successfully.
echo Make sure the computer is connected to the Internet, then run this file again.
echo If it still fails, send a screenshot of this window to ChatGPT.
echo ============================================================
pause

:end
endlocal
