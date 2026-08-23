@echo off
REM Copy each training run's final model into Assets so Unity can import it and you can
REM drop it onto a Behavior Parameters component in the Inspector.
REM
REM results/ deliberately lives OUTSIDE Assets/ - under Assets/ the editor generates a .meta
REM sibling for every file, including each events.out.tfevents.*, which breaks TensorBoard's
REM directory watcher. The tradeoff is that models are not visible to Unity until copied in.
REM This does that copy.
REM
REM Re-running overwrites in place, so the .meta and its GUID survive and any scene or prefab
REM already pointing at a model keeps working - you get the newer weights without re-assigning.
cd /d "%~dp0"

set DEST=Assets\Examples\Walker\TFModels
if not exist "%DEST%" mkdir "%DEST%"

echo Syncing final models into %DEST%
set COUNT=0
for /d %%R in (results\*) do (
    if exist "%%R\Walker.onnx" (
        copy /y "%%R\Walker.onnx" "%DEST%\%%~nxR.onnx" >nul
        echo    %%~nxR.onnx
        set /a COUNT+=1
    )
)

echo.
echo Done. Switch to Unity and let it regain focus to import them.
