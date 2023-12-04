# unitopatho baseline
python3 train_remix.py \
    --dataset Unitopatho \
    --model dsmil \
    --num_repeats 2 \
    --exp_name unitopatho_dsmil_baseline 

# unitopatho with one-prototype
python3 train_remix.py \
    --dataset Unitopatho \
    --model dsmil \
    --num_repeats 2 \
    --exp_name unitopatho_dsmil_K_1 \
    --num_prototypes 1