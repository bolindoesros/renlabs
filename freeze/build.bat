@echo off
rem Builds dist\Renlabs\Renlabs.exe (Windows). Set PYTHON to use another interpreter.
cd /d "%~dp0.."
if not defined PYTHON set PYTHON=.venv\Scripts\python.exe
"%PYTHON%" -m pip install -r freeze\requirements.txt || exit /b 1
"%PYTHON%" -m freeze.fetch_models || exit /b 1
"%PYTHON%" -m PyInstaller --noconfirm freeze\renlabs.spec || exit /b 1
