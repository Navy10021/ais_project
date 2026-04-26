@echo off
REM MCIS Pipeline Runner for Windows
REM ===============================

echo ======================================
echo MCIS Pipeline Runner
echo ======================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.11+
    pause
    exit /b 1
)

REM Create output directories
if not exist outputs\figures mkdir outputs\figures
if not exist outputs\tables mkdir outputs\tables
if not exist outputs\models mkdir outputs\models
if not exist outputs\reports mkdir outputs\reports

echo Running MCIS Pipeline...

REM Run full pipeline
python scripts\run_pipeline.py --step full

if errorlevel 1 (
    echo ERROR: Pipeline failed
    pause
    exit /b 1
)

echo ======================================
echo Pipeline Complete!
echo ======================================
echo.
echo Output directories:
echo   outputs\figures\ - Visualizations
echo   outputs\tables\   - Analysis tables
echo   outputs\models\   - Trained models
echo   outputs\reports\ - Final report

pause