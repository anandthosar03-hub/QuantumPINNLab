@echo off
if not exist .venv\Scripts\activate.bat (
  echo Virtual environment not found. Create it with: py -3.12 -m venv .venv
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m quantum_pinn.cli %*
