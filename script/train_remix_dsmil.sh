#!/bin/bash

dataset='C16_dataset_c16_low99_v0'
num_prototypes=$1 # {1, 2, 4, 8, 16}
num_epochs=200

# # ----------------------- Train MIL with ReMix features ------------------------- #
## --mode, choose among [None, replace, append, interpolate, cov, and joint]
## --rate, float number between [0, 1]
## For joint augmentation, we recommend a lower rate (e.g., 0.2 or 0.1) than our default rate of 0.5
## ReMix Paper Setting
##      - p = 0.5
##      - p = 0.1 for mode=joint

python train_mil_remix.py --dataset="${dataset}" \
    --model "dsmil" \
    --num_epochs $num_epochs\
    --mix_algorithm 'mixup' \
    --mix_domain 'feature' \
    --num_prototypes "${num_prototypes}" \
    --rate 0.5 \
    --num_classes=1 --gpu_index 0 

for mode in replace append interpolate cov joint
do
python train_mil_remix.py --dataset="${dataset}" \
    --model "dsmil" \
    --num_epochs $num_epochs\
    --mix_algorithm 'mixup' \
    --mix_domain 'feature' \
    --num_prototypes "${num_prototypes}" \
    --mode "${mode}" \
    --rate 0.5 \
    --num_classes=1 --gpu_index 0 
done

for rate in 0.1 0.2
do
python train_mil_remix.py --dataset="${dataset}" \
    --model "dsmil" \
    --num_epochs $num_epochs\
    --mix_algorithm 'mixup' \
    --mix_domain 'feature' \
    --num_prototypes "${num_prototypes}" \
    --mode joint \
    --rate "${rate}" \
    --num_classes=1 --gpu_index 0 
done


