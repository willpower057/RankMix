#!/bin/bash

# ------------------- DSMIL ------------------- #
python train_mil.py --dataset="C16_dataset_c16_low99_v0" --num_classes=1 --gpu_index 0 --num_workers 4 \
    --model "dsmil" --dropout_input 0.2



# ------------------- FRMIL ------------------- #
## The magnitude (i.e., --mag) for FRMIL used in this code follows the code from FRMIL
# different features will lead to different magnitude

python train_mil.py  --dataset="C16_dataset_c16_low99_v0" --num_classes=1 --gpu_index 0 \
    --model "frmil" \
    --loss "FrmilLoss"  \
    --mag 10.17 \
    --num_workers 4 \
    --dropout_input 0.2

