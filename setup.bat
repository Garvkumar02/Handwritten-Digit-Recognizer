@echo off
echo ============================================
echo   MNIST Digit Recognition - Setup Script
echo ============================================
echo.

REM Create virtual environment
echo [1/4] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

REM Install dependencies
echo [2/4] Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

REM Train the model
echo [3/4] Training CNN model (this may take a few minutes)...
python train_model.py

REM Launch the app
echo [4/4] Launching Streamlit app...
streamlit run app.py

pause
