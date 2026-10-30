@echo off
REM ===================================================================
REM  Patent Prior-Art Search  --  one-click launcher
REM
REM  Double-click this file to start the demo.
REM  If the project has not been built yet, it builds it first.
REM
REM  Uses the project's own virtual environment (.venv) when present, so
REM  it never depends on whatever is installed system-wide.
REM ===================================================================

cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo  No .venv found - falling back to the system Python.
    echo  To create one:  python -m venv .venv ^&^& .venv\Scripts\python -m pip install -r requirements.txt
    echo.
    set "PY=python"
)

echo.
echo  Patent Prior-Art Search
echo  Shivnaran S (22PD32) . Diravina E I (22PD12)
echo  -------------------------------------------------
echo.

if not exist "data\corpus.jsonl" (
    echo  First run detected - building the project.
    echo  This downloads the dataset and the models, and trains the rerankers.
    echo  Expect about two hours. Press Ctrl+C to cancel.
    echo.
    "%PY%" scripts\setup.py
    if errorlevel 1 (
        echo.
        echo  Build failed. See the error above.
        pause
        exit /b 1
    )
)

echo  Starting the app at http://localhost:8501
echo  Close this window to stop it.
echo.
"%PY%" -m streamlit run app.py

pause
