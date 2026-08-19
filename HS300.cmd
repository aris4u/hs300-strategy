@echo off
title HS300
cd /d "%~dp0"

set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not exist "%PY%" if exist "%~dp0runtime\python.exe" set "PY=%~dp0runtime\python.exe"
if not exist "%PY%" if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

curl.exe -s -o nul --connect-timeout 1 http://127.0.0.1:8765/ >nul 2>&1
if not errorlevel 1 (
  echo 界面已在运行，直接打开。
  start "" "http://127.0.0.1:8765/"
  exit /b 0
)

if exist "%~dp0hs300_strategy\integrity.py" (
  "%PY%" "%~dp0hs300_strategy\integrity.py"
  if errorlevel 1 (
    pause
    exit /b 1
  )
)

echo 正在打开选股界面...
"%PY%" "%~dp0app.py"
if errorlevel 1 (
  echo.
  echo 启动失败，窗口不会马上关掉。
  pause
)
