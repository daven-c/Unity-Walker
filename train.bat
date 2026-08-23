@echo off
REM Always run from the repo root, regardless of where this was launched from,
REM so mlagents-learn's --results-dir resolves to the same place every time.
cd /d "%~dp0"

if not exist venv\Scripts\activate.bat (
    echo venv not found at %~dp0venv - create it first, see README.md
    exit /b 1
)

REM results/ deliberately lives OUTSIDE Assets/. Under Assets/ Unity's asset pipeline
REM generates a .meta sibling for every file, including each events.out.tfevents.*,
REM which breaks TensorBoard's directory watcher (it sorts alphabetically and latches
REM onto the .meta) and pointlessly imports 16 MB .pt checkpoints as Unity assets.
call venv\Scripts\activate.bat
mlagents-learn Assets\Python\config.yaml --results-dir=results %*
