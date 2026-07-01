@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title E.D.I.T.H. - Intelligence Framework
cd /d "%~dp0"

echo.
echo  ============================================================
echo    E.D.I.T.H. - Starting All Systems
echo  ============================================================
echo.

:: -- Clear stale Python bytecode cache --------------------------
echo [EDITH] Clearing Python cache...
for /d /r . %%d in (__pycache__) do (
    if exist "%%d" rd /s /q "%%d" 2>nul
)
del /s /q *.pyc >nul 2>nul

:: -- Locate Python ------------------------------------------------
set "PYTHON="
for %%P in (python python3 py) do (
    if not defined PYTHON (
        %%P --version >nul 2>&1
        if not errorlevel 1 set "PYTHON=%%P"
    )
)
if not defined PYTHON (
    color 0C
    echo.
    echo  [ERROR] Python not found. Install Python 3.10+ from https://python.org
    echo  Make sure to check "Add Python to PATH" during install.
    echo.
    goto :end
)

:: -- Confirm main.py exists ---------------------------------------
if not exist "main.py" (
    color 0C
    echo.
    echo  [ERROR] main.py not found in this folder: %cd%
    echo  Move run.bat into the same folder as main.py and try again.
    echo.
    goto :end
)

:: -- Launch ---------------------------------------------------------
echo [EDITH] Launching with !PYTHON!...
echo.
!PYTHON! main.py
if errorlevel 1 (
    color 0C
    echo.
    echo  [ERROR] E.D.I.T.H. exited with an error. See the messages above.
)

:end
echo.
echo  ------------------------------------------------------------
echo  Press any key to close this window...
pause >nul
