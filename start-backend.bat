@echo off
cd backend
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo ==========================================
echo   Video Insights App
echo   Open http://127.0.0.1:8000 in browser
echo ==========================================
echo.
python manage.py runserver
