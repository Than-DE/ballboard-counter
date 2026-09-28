@echo off
rem Builds dist\BallBoardCounter.exe on Windows.
rem Needs Python 3.10+ from python.org (installer option "Add python.exe to PATH").
cd /d "%~dp0"
py -3 -m venv .venv || goto :error
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt pyinstaller || goto :error
.venv\Scripts\python test_counter.py || goto :error
.venv\Scripts\pyinstaller --noconfirm --onefile --windowed --name BallBoardCounter --hidden-import natnet app.py || goto :error
echo.
echo Done: dist\BallBoardCounter.exe
pause
exit /b 0
:error
echo Build failed.
pause
exit /b 1
