# RankMix-Data-Augmentation-for-Weakly-Supervised-Learning-of-Classifying-Whole-Slide-Images

# Requirements
## 1. Install the required package for training model (It may takes several hours to search the dependency):
```shell
conda env create --name rankmix --file env.yml
conda activate rankmix
```

## 2. If you want to crop the slide and produce the feature by yourself, the "openslide" package is required:
  
```shell
sudo apt-get install openslide-tools
pip install openslide-python
```
<!-- #### If you encounter "OSError: libopenslide.so.0: cannot open shared object file: No such file or directory"
```shell
sudo apt-get install libopenslide0
``` -->

## 3. If you want to run the code of Remix:
```shell
conda install -c pytorch faiss-gpu
```


# Get our patch features and model weights
1. patch features of Camelyon16 dataset
2. model weights of feature extractor for Camelyon16
To be updated



# Preprocessing raw WSI from scratch
## Raw data
- you can download [Camelyon16](https://camelyon17.grand-challenge.org/Data) dataset. 

## Process dataset
You may need to change some folder path in the following steps.
### Crop the slide into patches and compute the features (More detail can be found in [DSMIL](https://github.com/binli123/dsmil-wsi))
``` shell
sh script/preprocessing.sh
```

### Convert the dataset from .csv into .npy (speed up training) and prepare the reduced feature for ReMix (More detail can be found in [ReMix](https://github.com/Jiawei-Yang/ReMix))
``` shell
sh script/prepare_remix.sh
```

The magnitude used by FRMIL follows the code from [FRMIL](https://github.com/PhilipChicco/FRMIL)

# Training
## 1. Train the DSMIL/FRMIL model:
``` shell
sh script/train_mil.sh
```

## 2. Train the DSMIL/FRMIL model with RankMix:
``` shell
sh script/train_rankmix.sh
```

## 3. Train the DSMIL/FRMIL model with ReMix:
``` shell
sh script/train_remix_dsmil.sh 1
sh script/train_remix_dsmil.sh 2
sh script/train_remix_dsmil.sh 4
sh script/train_remix_dsmil.sh 8
sh script/train_remix_dsmil.sh 16

sh script/train_remix_frmil.sh 1
sh script/train_remix_frmil.sh 2
sh script/train_remix_frmil.sh 4
sh script/train_remix_frmil.sh 8
sh script/train_remix_frmil.sh 16
```

#### Select the final result of Remix
```shell
python select_remix.py
```

# References
1. https://github.com/binli123/dsmil-wsi
2. https://github.com/PhilipChicco/FRMIL
3. https://github.com/Jiawei-Yang/ReMix
4. https://github.com/agaldran/balanced_mixup


# Citation
```
@inproceedings{chen2023rankmix,
  title={RankMix: Data Augmentation for Weakly Supervised Learning of Classifying Whole Slide Images With Diverse Sizes and Imbalanced Categories},
  author={Yuan-Chih Chen and Chun-Shien Lu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23936--23945},
  year={2023}
}
```