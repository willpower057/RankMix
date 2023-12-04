import warnings
warnings.filterwarnings("ignore")

import torch, random 
import torch.nn as nn
import torch, os, glob, sys, numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from PIL import Image


class FrmilClassDataset(Dataset):    
    def __init__(self, df=None, mode='train', batch=None, num_classes=2):   
        self.df = df
        if isinstance(df, list):
            self.df = [line.strip().split(',') for line in self.df]
            self.df = pd.DataFrame(self.df)
            self.df.iloc[:, 1] = self.df.iloc[:, 1].astype('int')            
        self.split = mode        
        self.num_classes = num_classes
        self.batch = batch
        
        self.ndims = 512 
        
        self.pos_weight, self.count_dict, self.labels = self.computeposweight()
        print("Total bags:", self.__len__())
        if self.batch:            
            self.bag_mu, self.bag_max, self.bag_min, self.bag_sum = self.get_bag_sizes()
            print(f'mu {self.bag_mu} | min {self.bag_min} | max {self.bag_max} | total {self.bag_sum}\n')
        
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()
        
        row_data = self.df.iloc[index]
        path = os.path.join('..', 'dsmil-wsi', row_data.iloc[0])
        if path.endswith('csv'):
            bag = pd.read_csv(path).to_numpy()
        else:
            bag = np.load(path)        
        target = row_data.iloc[1]

        wsi_id  = index   

        target = np.zeros(self.num_classes)
        if self.num_classes==1:
            target[0] = row_data.iloc[1]
        else:
            if int(row_data.iloc[1])<=(len(target)-1):
                target[int(row_data.iloc[1])] = 1

        if self.batch:
            num_inst  = bag.shape[0] 
            # bag_feats = np.zeros((self.bag_max,self.ndims),dtype=np.float) 
            bag_feats = np.zeros((self.bag_max,self.ndims),dtype=float) # NumPy>=1.24, remove np.float            
            bag = np.asarray(bag) 
            bag_feats[:num_inst,:] = bag   
            return torch.from_numpy(bag_feats).float(), target, [wsi_id], num_inst
        else:

            return torch.from_numpy(np.asarray(bag)).float(), torch.from_numpy(target), [wsi_id]

    def __len__(self):
        return len(self.df)
    
    # compute postive weights for loss + class counts 
    def computeposweight(self):
        pos_count  = 0
        if self.num_classes==1:
            count_dict = {x: 0 for x in range(2)}
        else:
            count_dict = {x: 0 for x in range(self.num_classes)}

        labels     = []

        for item in range(len(self.df)):
            cls_id = self.df.iloc[item][1] 
            pos_count += cls_id
            count_dict[cls_id] += 1
            labels.append(cls_id)
        return torch.tensor((len(self.df)-pos_count)/pos_count), count_dict, labels
    
    def get_bag_sizes(self):
        bags = []
        for item in range(len(self.df)):
            path = os.path.join('..', 'dsmil-wsi', self.df.iloc[item].iloc[0])
            if path.endswith('csv'):
                feats = pd.read_csv(path).to_numpy()
            else:
                feats = np.load(path)


            num_insts = np.asarray(feats).shape[0]
            bags.append(num_insts)
        return np.mean(bags),np.max(bags), np.min(bags), np.sum(bags)
   
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()
        row_data = self.df.iloc[index]
        path = os.path.join('..', 'dsmil-wsi', row_data.iloc[0])

        bag = np.load(path, allow_pickle=True)
        target = row_data.iloc[1]

        wsi_id  = index  

        target = np.zeros(self.num_classes)
        if self.num_classes==1:
            target[0] = row_data.iloc[1]
        else:
            if int(row_data.iloc[1])<=(len(target)-1):
                target[int(row_data.iloc[1])] = 1

        if self.batch:
            num_inst  = bag.shape[0] 
            bag_feats = np.zeros((self.bag_max,self.ndims),dtype=np.float)
            bag = np.asarray(bag) 
            bag_feats[:num_inst,:] = bag    
            return torch.from_numpy(bag_feats).float(), target, [wsi_id], num_inst
        else:            
            return torch.from_numpy(np.asarray(bag)).float(), torch.from_numpy(target), [wsi_id]

    def __len__(self):
        return len(self.df)
    
    # compute postive weights for loss + class counts 
    def computeposweight(self):
        pos_count  = 0
        if self.num_classes==1:
            count_dict = {x: 0 for x in range(2)}
        else:
            count_dict = {x: 0 for x in range(self.num_classes)}

        labels     = []
        for item in range(len(self.df)):
            cls_id = self.df.iloc[item][1] 
            pos_count += cls_id
            count_dict[cls_id] += 1
            labels.append(cls_id)
        return torch.tensor((len(self.df)-pos_count)/pos_count), count_dict, labels
    
    def get_bag_sizes(self):
        bags = []
        for item in range(len(self.df)):
            path = os.path.join('..', 'dsmil-wsi', self.df.iloc[item].iloc[0])
            feats = np.load(path, allow_pickle=True)

            num_insts = np.asarray(feats).shape[0]
            bags.append(num_insts)
        return np.mean(bags),np.max(bags), np.min(bags), np.sum(bags)

class RemixToFrmilClassDataset(Dataset):
    
    def __init__(self, feats, labels, mode='train', batch=None, num_classes=2):
        self.feats = feats
        self.labels = labels
        self.split = mode
        self.num_classes = num_classes
        self.batch = batch
        
        self.ndims = 512 
      
        self.pos_weight, self.count_dict, self.labels = self.computeposweight()
        print("Total bags:", self.__len__())
        if self.batch:            
            self.bag_mu, self.bag_max, self.bag_min, self.bag_sum = self.get_bag_sizes()
            print(f'mu {self.bag_mu} | min {self.bag_min} | max {self.bag_max} | total {self.bag_sum}\n')
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()

        if self.split == 'train':            
            bag = self.feats[index]
            label = self.labels[index]
        else:   
            bag_path, label = self.feats[index].strip().split(',')
            bag = np.load(bag_path) 
            y = int(label)

        wsi_id  = index   
        target = np.zeros(self.num_classes)
        if self.num_classes==1:
            target[0] = label
        else:
            if int(label)<=(len(target)-1):
                target[int(label)] = 1

        if self.batch:
            num_inst  = bag.shape[0] 
            # bag_feats = np.zeros((self.bag_max,self.ndims),dtype=np.float) # valid for older numpy version
            bag_feats = np.zeros((self.bag_max,self.ndims),dtype=float)
            bag = np.asarray(bag) 
            bag_feats[:num_inst,:] = bag            

            return torch.from_numpy(bag_feats).float(), target, [wsi_id], num_inst
        else:
            return torch.from_numpy(np.asarray(bag)).float(), torch.from_numpy(target), [wsi_id]

    def __len__(self):
        return len(self.feats)
    
    # compute postive weights for loss + class counts 
    def computeposweight(self):
        pos_count  = 0
        if self.num_classes==1:
            count_dict = {x: 0 for x in range(2)}
        else:
            count_dict = {x: 0 for x in range(self.num_classes)}

        labels = []      
        for item in range(len(self.feats)):
            cls_id = self.labels[item]
            pos_count += cls_id
            count_dict[cls_id] += 1
            labels.append(cls_id)
        return torch.tensor((len(self.feats)-pos_count)/pos_count), count_dict, labels
    
    def get_bag_sizes(self):        
        n_slides = self.feats.shape[0]
        k = self.feats.shape[1]
        
        return k, k, k, n_slides*k