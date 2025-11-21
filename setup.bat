@echo off
echo ==========================================
echo Universal Media Enhancer - Setup Script
echo ==========================================
echo.

REM Create directory structure
echo Creating directory structure...
mkdir src\image_processing 2>nul
mkdir src\signal_processing 2>nul
mkdir src\metrics 2>nul
mkdir output 2>nul
mkdir data\sample_images 2>nul
mkdir data\sample_audio 2>nul

REM Create __init__.py files
echo Creating Python package files...
type nul > src\__init__.py
type nul > src\image_processing\__init__.py
type nul > src\signal_processing\__init__.py
type nul > src\metrics\__init__.py

REM Install dependencies
echo.
echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Copy your DSP/DIP toolkit modules to src\
echo 2. Add sample data to data\ folder (optional)
echo 3. Run: python main.py
echo.
pause
