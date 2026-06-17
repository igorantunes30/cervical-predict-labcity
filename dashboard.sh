#!/bin/bash
# Dashboard web para monitoramento do treino em tempo real
# Acesse: http://localhost:7860

PYTHON="/mnt/c/Users/igor_/miniconda3/envs/RNA/python.exe"
cd "$(dirname "$0")"
echo "Iniciando dashboard em http://localhost:7860 ..."
$PYTHON dashboard.py
