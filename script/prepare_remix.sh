#!/bin/bash

# # ----------------------- Prepare data ------------------------- #
## 1. Create Dataset with Remix's format
# python tools/process_dataset.py --dataset Camelyon16 --task download convert split
python ReMix/tools/process_dataset.py --dataset rankmix_c16 --task convert split


# # ----------------------- Use KMeans to produce reduced features ------------------------- #
## --num_prototypes {1, 2, 4, 8, 16}
## C16_dataset_c16_low99_v0: 32
for k in 1 2 4 8 16
do
python ReMix/reduce.py --dataset C16_dataset_c16_low99_v0 --num_prototypes "${k}"
done