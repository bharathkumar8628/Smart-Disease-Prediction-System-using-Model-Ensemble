@echo off
echo ===================================================
echo Starting Smart Disease Prediction Web Application...
echo ===================================================
call .\venv\Scripts\activate.bat
streamlit run src\app.py
pause
