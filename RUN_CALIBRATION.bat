@echo off
REM ============================================================
REM Quick Start Script untuk Kalibrasi Depth Anything V2
REM ============================================================

echo.
echo ========================================
echo  KALIBRASI DEPTH ANYTHING V2
echo ========================================
echo.

REM Aktifkan virtual environment
if exist ".venv\Scripts\activate.bat" (
    echo [1/2] Mengaktifkan virtual environment...
    call .venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment tidak ditemukan!
    echo Pastikan folder .venv ada di direktori ini.
    pause
    exit /b 1
)

REM Jalankan kalibrasi
echo.
echo [2/2] Menjalankan kalibrasi...
echo.
python conversion/start_calibration.py

echo.
pause
