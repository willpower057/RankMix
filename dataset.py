import os.path as osp
import copy
from stringprep import in_table_d2
import sys, os, glob
import numbers
import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as tr
import torchvision.transforms.functional as VF
from sklearn.utils import shuffle

class ClassDataset(Dataset):
    def __init__(self, data, args, transforms=None, test=False):
        self.data = data
        self.args = args            
        self.transforms = transforms

    def __getitem__(self, index):              
        csv_file_df = self.data.iloc[index]
        if self.args.dataset == 'TCGA-lung-default':
            feats_path = 'datasets/tcga-dataset/tcga_lung_data_feats-npy/' + csv_file_df.iloc[0].split('/')[1] + '.npy'
        else:            
            feats_path = os.path.join(
                os.path.dirname(csv_file_df.iloc[0]) + '-npy', 
                os.path.basename(csv_file_df.iloc[0]).replace('.csv', '.npy')
            )
        y = int(csv_file_df.iloc[1])
        
        feats = np.load(feats_path)            
        label = np.zeros(self.args.num_classes)
        
        if self.args.num_classes==1:
            label[0] = y
        else:
            if y<=(len(label)-1):
                label[y] = 1
        if self.transforms is not None:
            img = self.transforms(img)
        
        return torch.Tensor(label), torch.Tensor(feats)

    def __len__(self):
        return len(self.data)

class ClassDatasetRemix(Dataset):
    def __init__(self, feats, labels, args, mode='train', transforms=None, test=False):        
        self.feats = feats
        self.labels = labels
        self.args = args
        self.mode = mode            
        self.transforms = transforms

        if self.mode == 'train':
            self.feats = torch.Tensor(self.feats)
            self.labels = torch.LongTensor(self.labels)
        

    def __getitem__(self, index):
        # load image and labels 
        if self.mode == 'train':
            feat = self.feats[index]
        else:            
            feat = np.load(self.feats[index].split(',')[0])
          
        label = np.zeros(self.args.num_classes)
        if self.args.num_classes==1:
            label[0] = self.labels[index]
        else:
            if int(self.labels[index])<=(len(label)-1):
                label[int(self.labels[index])] = 1
        if self.transforms is not None:
            img = self.transforms(img)        
        return torch.Tensor(label), torch.Tensor(feat)

    def __len__(self):
        return len(self.feats)

# https://github.com/huanghoujing/pytorch-wrapping-multi-dataloaders/blob/master/wrapping_multi_dataloaders.py
class ComboIter(object):
    """An iterator."""
    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [next(loader_iter) for loader_iter in self.loader_iters]
        # batches = [loader_iter.next() for loader_iter in self.loader_iters] # different torch version
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)

class ComboLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.
    Args:
    loaders: a list or tuple of pytorch DataLoader objects
    """
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return ComboIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches

def get_sampling_probabilities(class_count, mode='instance', ep=None, n_eps=None):
    '''
    Note that for progressive sampling I use n_eps-1, which I find more intuitive.
    If you are training for 10 epochs, you pass n_eps=10 to this function. Then, inside
    the training loop you would have sth like 'for ep in range(n_eps)', so ep=0,...,9,
    and all fits together.
    '''
    if mode == 'instance':
        q = 0
    elif mode == 'class':
        q = 1
    elif mode == 'sqrt':
        q = 0.5 # 1/2
    elif mode == 'cbrt':
        q = 0.125 # 1/8
    elif mode == 'prog':
        assert ep != None and n_eps != None, 'progressive sampling requires to pass values for ep and n_eps'
        relative_freq_imbal = class_count ** 0 / (class_count ** 0).sum()
        relative_freq_bal = class_count ** 1 / (class_count ** 1).sum()
        sampling_probabilities_imbal = relative_freq_imbal ** (-1)
        sampling_probabilities_bal = relative_freq_bal ** (-1)
        return (1 - ep / (n_eps - 1)) * sampling_probabilities_imbal + (ep / (n_eps - 1)) * sampling_probabilities_bal
    else: sys.exit('not a valid mode')

    relative_freq = class_count ** q / (class_count ** q).sum()
    sampling_probabilities = relative_freq ** (-1)

    return sampling_probabilities

def modify_loader(loader, mode, dataset_type=None, ep=None, n_eps=None):    
    label_col = loader.dataset.data.iloc[:,1]
    class_count = np.unique(label_col, return_counts=True)[1]
    sampling_probs = get_sampling_probabilities(class_count, mode=mode, ep=ep, n_eps=n_eps)
    sample_weights = sampling_probs[label_col]

    mod_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))
    mod_loader = DataLoader(loader.dataset, batch_size = loader.batch_size, sampler=mod_sampler, num_workers=loader.num_workers)
    return mod_loader

def get_combo_loader(loader, base_sampling='instance', dataset_type=None):
    if base_sampling == 'instance':
        imbalanced_loader = loader
    else:
        imbalanced_loader = modify_loader(loader, mode=base_sampling, dataset_type=dataset_type)

    balanced_loader = modify_loader(loader, mode='class', dataset_type=dataset_type)
    combo_loader = ComboLoader([imbalanced_loader, balanced_loader])
    return combo_loader