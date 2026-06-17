@echo off
REM Dashboard web para monitoramento do treino em tempo real
REM Acesse: http://localhost:7860

set PYTHON=C:\Users\igor_\miniconda3\envs\RNA\python.exe

echo Iniciando dashboard em http://localhost:7860 ...
echo Pressione Ctrl+C para parar.
echo.

%PYTHON% dashboard.py
pause
