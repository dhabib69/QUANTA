@echo off
title QUANTA v11.5b
cd /d "%~dp0"

:: Wait for bot to start Flask, then open browser
start "" cmd /c "ping -n 61 127.0.0.1 >nul && start http://127.0.0.1:5000"

"C:\Users\habib\AppData\Local\Programs\Python\Python311\python.exe" main.py
pause
