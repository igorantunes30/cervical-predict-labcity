@echo off
REM GLSim - Classificacao de Cancer de Colo Uterino (Herlev Dataset)
REM 5 classes: Dyskeratotic, Koilocytotic, Metaplastic, Parabasal, Superficial-Intermediate
REM
REM Uso: train.bat [lr] [epochs] [serial]
REM   train.bat                  -> lr=0.003, 50 epocas
REM   train.bat 0.001 100 1
REM   train.bat 0.003 50 0 --train_trainval   (treina em train+val, avalia no test)

set LR=%1
if "%LR%"=="" set LR=0.003

set EPOCHS=%2
if "%EPOCHS%"=="" set EPOCHS=50

set SERIAL=%3
if "%SERIAL%"=="" set SERIAL=0

set PYTHON=C:\Users\igor_\miniconda3\envs\RNA\python.exe

echo ============================================
echo  GLSim - Cervical Cancer Classification
echo  LR=%LR%  EPOCHS=%EPOCHS%  SERIAL=%SERIAL%
echo  Dataset: Herlev (5 classes, 4049 celulas)
echo ============================================

%PYTHON% tools/train.py ^
    --cfg configs/train.yaml ^
    --cfg_method configs/methods/glsim.yaml ^
    --model_name vit_b16 ^
    --classifier cls ^
    --anchor_size 16 ^
    --lr %LR% ^
    --epochs %EPOCHS% ^
    --serial %SERIAL% ^
    --batch_size 16 ^
    --eval_freq 1 ^
    --save_freq 50

echo.
echo Treino finalizado! Modelo salvo em results_train/
pause
