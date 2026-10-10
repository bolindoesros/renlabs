@echo off
rem Windows launcher; no venv activation needed.
cd /d "%~dp0"
".venv\Scripts\python.exe" -m ren.ui %*
