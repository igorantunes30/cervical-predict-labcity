@echo off
REM Inicia TensorBoard apontando para os logs de treino
REM Acesse: http://localhost:6006

set TB=C:\Users\igor_\miniconda3\envs\RNA\Scripts\tensorboard.exe

echo Iniciando TensorBoard em http://localhost:6006 ...
echo Pressione Ctrl+C para parar.
echo.

"%TB%" --logdir "D:\mestrado\RNA\SEMINARIO\cervical-cancer\results_train" --host 0.0.0.0 --port 6006
pause
