#!/bin/bash

### 1. ---------- Prepare the patches ----------
# The quantity of patches in different settings, we use=1,3 which follows DSMIL's setting
# "C16_dataset/training" (270 slides): m=1 | 4,143,792
# "C16_dataset/training" (270 slides): m=2 | 1,331,924 
# "C16_dataset/training" (270 slides): m=3 |   365,360 
# "C16_dataset/testing" (129 slides): m=1  | 1,884,528
# "C16_dataset/testing" (129 slides): m=2  |   614,798 
# "C16_dataset/testing" (129 slides): m=3  |   167,458 

dataset_dir='dataset'
python deepzoom_tiler.py \
    --dataset_dir ${dataset_dir} \
    --dataset "C16_dataset/training" \
    --slide_format "tif" \
    -m 1 3 \
    --base_mag 20 \
    --tile_size 224 \
    --overlap 0 \
    --workers 16

python deepzoom_tiler.py \
    --dataset_dir ${dataset_dir} \
    --dataset "C16_dataset/testing" \
    --slide_format "tif" \
    -m 1 3 \
    --base_mag 20 \
    --tile_size 224 \
    --overlap 0 \
    --workers 16

## 2. ---------- Train the embedder ----------
# dataset="C16_dataset/training"
# python simclr/run.py --multiscale=1 --level=low --dataset_dir ${dataset_dir} --dataset ${dataset}
# python simclr/run.py --multiscale=1 --level=high --dataset_dir ${dataset_dir} --dataset ${dataset}

## 3. ---------- Compute the features ----------
# python compute_feats.py \
#     --dataset="C16_dataset/training" \
#     --magnification=tree \
#     --dataset_dir ${dataset_dir} \
#     --output_dir "C16_dataset_c16_low99_v0/training" \
#     --weights_high='mil_c16_v0' \
#     --weights_low='c16_low_99' \
#     --batch_size 128 \
#     --num_workers 4 \
#     --gpu_index 0 \
#     --use_cuda

# python compute_feats.py \
#     --dataset="C16_dataset/testing" \
#     --magnification=tree \
#     --dataset_dir ${dataset_dir} \
#     --output_dir "C16_dataset_c16_low99_v0/testing" \
#     --weights_high='mil_c16_v0' \
#     --weights_low='c16_low_99' \
#     --batch_size 128 \
#     --num_workers 4 \
#     --gpu_index 0 \
#     --use_cuda
