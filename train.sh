#!/bin/bash
# GLSim - Classificacao de Cancer de Colo Uterino (Herlev Dataset)
# 5 classes: Dyskeratotic, Koilocytotic, Metaplastic, Parabasal, Superficial-Intermediate
#
# Uso: bash train.sh [lr] [epochs] [serial]
#   bash train.sh               -> lr=0.003, 50 epocas
#   bash train.sh 0.001 100 1

LR="${1:-0.003}"
EPOCHS="${2:-50}"
SERIAL="${3:-0}"

PYTHON="/mnt/c/Users/igor_/miniconda3/envs/RNA/python.exe"

echo "============================================"
echo " GLSim - Cervical Cancer Classification"
echo " LR=$LR  EPOCHS=$EPOCHS  SERIAL=$SERIAL"
echo " Dataset: Herlev (5 classes, 4049 celulas)"
echo "============================================"

cd "$(dirname "$0")"

$PYTHON tools/train.py \
    --cfg configs/train.yaml \
    --cfg_method configs/methods/glsim.yaml \
    --model_name vit_b16 \
    --classifier cls \
    --anchor_size 16 \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --serial "$SERIAL" \
    --batch_size 16 \
    --eval_freq 1 \
    --save_freq 50
