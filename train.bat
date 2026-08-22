@echo off
REM Always run from the repo root, regardless of where this was launched from,
REM so mlagents-learn's --results-dir resolves to the same place every time.
cd /d "%~dp0"

if not exist venv\Scripts\activate.bat (
    echo venv not found at %~dp0venv - create it first, see README.md
    exit /b 1
)

call venv\Scripts\activate.bat
mlagents-learn Assets\Python\config.yaml --results-dir=Assets\Python\results %*
