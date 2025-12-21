#!/bin/bash
set -e

PROJECT_DIR="$HOME/projects/htr_trainer"

cd "$PROJECT_DIR"
source venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

python3 ./scripts/train.py \
    --epochs 20 \
    --batch-size 6 \
    --eval-batch-size 2