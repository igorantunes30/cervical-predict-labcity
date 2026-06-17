#!/bin/bash
PYTHON=/mnt/c/Users/igor_/miniconda3/envs/RNA/python.exe

$PYTHON tools/train.py \
    --cfg configs/train_slides.yaml \
    --cfg_method configs/methods/glsim.yaml \
    --model_name vit_b16 \
    --classifier cls \
    --anchor_size 16 \
    --lr 0.003 \
    --epochs 50 \
    --serial 1 \
    --batch_size 16 \
    --eval_freq 1 \
    --save_freq 50
