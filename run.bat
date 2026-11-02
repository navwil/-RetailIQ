@echo off
echo ========================================================
echo Launching RetailIQ Enterprise Analytics Platform...
echo ========================================================
if not exist venv (
    echo Creating Python Virtual Environment...
    python -m venv venv
)
call venv\Scripts\activate.bat
echo Installing dependencies...
pip install -r requirements.txt
echo Starting Streamlit Server...
streamlit run app.py
pause
