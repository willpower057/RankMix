#!/bin/bash

## 1. Please use the model trained by train_mil.sh as a pre-trained model.
# Then, modify the "load_model" function in utils.py to define the weight used by train_mil_rankmix.py

## 2. Different mix_strategy you can try:
# traditional mixup approaches: [mix direct, duplicated2, shrink, sampling, uniform, uniform_partial, duplicated]
# traditional cutmix approaches: [uniform, cutpaste, cutpaste2]
# rankmix approach: [rank, rank2, rank_90, rank2_90]

# set mix_algorithm='mixup' for mixup and rankmix approaches
# set mix_algorithm='cutmix' for cutmix approaches

## 3. top_k: ['min' or int]


# ------------------- DSMIL ------------------- #
# C16_dataset_c16_low99_v0
rank_strategy=rank
python train_mil_rankmix.py --dataset="C16_dataset_c16_low99_v0" \
    --model "dsmil" \
    --mix_algorithm 'mixup' \
    --mix_domain 'feature' \
    --mix_strategy "${rank_strategy}" \
    --num_classes=1 --gpu_index 0 \
    --fix_score_function 'finetune'\
    --pretrain

# ------------------- FR-MIL ------------------- #
# (C16_dataset_c16_low99_v0, 10.17, num_classes=1)
rank_strategy=rank
python train_mil_rankmix.py --dataset="C16_dataset_c16_low99_v0" \
    --model "frmil" \
    --loss "FrmilLoss"  \
    --mag 10.17 \
    --mix_algorithm 'mixup' \
    --mix_domain 'feature' \
    --mix_strategy "${rank_strategy}" \
    --num_classes=1 --gpu_index 0\
    --num_workers 4 \
    --fix_score_function 'finetune'\
    --pretrain
    
    
      
