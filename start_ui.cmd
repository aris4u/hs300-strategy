@echo off
setlocal EnableDelayedExpansion
title HS300
cd /d "%~dp0"

set "PY="
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"

echo 正在启动选股界面...
if defined PY (
  "%PY%" app.py
) else (
  py -3 app.py
)
if errorlevel 1 (
  echo.
  echo 启动失败，窗口不会马上关掉。
  pause
)
endlocal
