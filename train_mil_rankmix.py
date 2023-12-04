from re import S
import sys, argparse, os, copy, itertools, glob, datetime
from collections import OrderedDict
import time
from tkinter.messagebox import YES
from typing import Tuple
from random import randint, choice
from xmlrpc.client import boolean

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, classification_report
from sklearn.metrics import auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file 

from dataset import ClassDataset, get_sampling_probabilities, get_combo_loader
from dataset_frmil import FrmilClassDataset
from train_mil import FeatMag
from samplers import CategoriesSampler
from utils import load_model, print_result

def mixup_features(features_pair: Tuple[torch.Tensor, torch.Tensor],
        lam: float,
        mix_strategy,                  
        ) -> torch.Tensor:
    features1, features2 = features_pair
    n_patch1, n_patch2 = features1.shape[0], features2.shape[0]
    feature_dim = features1.shape[1:]
    max_patch = max(n_patch1, n_patch2)   
    min_patch = min(n_patch1, n_patch2)     
    
    if mix_strategy=='direct' or mix_strategy.startswith('rank'):
        if n_patch1>n_patch2:
            n_pad = n_patch1 - n_patch2       
            pad_features2 = F.pad(features2, (0, 0, 0, n_pad), "constant", 0)
            return (1 - lam) * features1 + lam * pad_features2
        else: 
            n_pad = n_patch2 - n_patch1       
            pad_features1 = F.pad(features1, (0, 0, 0, n_pad), "constant", 0)
            return (1 - lam) * pad_features1 + lam * features2

    elif mix_strategy=='sampling':
        sampling_probs = get_sampling_probabilities(torch.Tensor([n_patch1, n_patch2]).cuda(), mode='instance', ep=None, n_eps=None)
        sampling_weights = torch.cat([sampling_probs[0].repeat(n_patch1), sampling_probs[1].repeat(n_patch2)])
        ids = torch.multinomial(sampling_weights, max_patch, replacement=True).view(max_patch, -1)
        idx = ids.repeat(1, *feature_dim)
        imbalaced_features = torch.gather(torch.cat(features_pair, 0), 0, idx)

        sampling_probs = get_sampling_probabilities(torch.Tensor([n_patch1, n_patch2]).cuda(), mode='class', ep=None, n_eps=None)
        sampling_weights = torch.cat([sampling_probs[0].repeat(n_patch1), sampling_probs[1].repeat(n_patch2)])
        ids = torch.multinomial(sampling_weights, max_patch, replacement=True).view(max_patch, -1)
        idx = ids.repeat(1, *feature_dim) 
        balaced_features = torch.gather(torch.cat(features_pair, 0), 0, idx)  
        return (1 - lam) * imbalaced_features + lam * balaced_features

    elif mix_strategy=='uniform': # sample N=max_patch features from two slides independently.
        ids_1 = torch.randint(0, n_patch1, (max_patch,))        
        ids_2 = torch.randint(0, n_patch2, (max_patch,))        
        return (1 - lam) * features1[ids_1] + lam * features2[ids_2]
    
    elif mix_strategy=='uniform_partial': # sample N=max_patch features only from the smaller slide.        
        small_index = [n_patch1, n_patch2].index(min_patch)

        ids = torch.randint(0, min_patch, (max_patch,))
        small_features = features_pair[small_index]
        sampled_features = small_features[ids]

        if small_index==0:
            return (1 - lam) * sampled_features + lam * features2  
        elif small_index==1:
            return (1 - lam) * features1 + lam * sampled_features  

        raise NotImplementedError  

    elif mix_strategy=='duplicated':
        big_index = [n_patch1, n_patch2].index(max_patch)
        small_index = 0 if big_index==1 else 1        

        big_features = features_pair[big_index]
        small_features = features_pair[small_index]
        
        multiplier = big_features.shape[0] // small_features.shape[0]
        remainder = big_features.shape[0] % small_features.shape[0]
        
        resized_features = torch.cat((small_features.repeat(multiplier, 1), small_features[:remainder]),0)
        if small_index==0:
            return (1 - lam) * resized_features + lam * big_features  
        elif small_index==1:
            return (1 - lam) * big_features + lam * resized_features
    elif mix_strategy=='duplicated2':
        # resize feature1 to fit feature 2
        if n_patch1<n_patch2: 
            multiplier = n_patch2 // n_patch1
            remainder = n_patch2 % n_patch1
            resized_features = torch.cat((features1.repeat(multiplier, 1), features1[:remainder]),0)
            return (1 - lam) * resized_features + lam * features2
        else: 
            start_idx = randint(0, n_patch1-n_patch2)
            shrink_features = features1[start_idx:start_idx+n_patch2]
            return (1 - lam) * shrink_features + lam * features2        

    elif mix_strategy=='shrink':
        big_index = [n_patch1, n_patch2].index(max_patch)
        small_index = 0 if big_index==1 else 1        

        big_features = features_pair[big_index]
        small_features = features_pair[small_index]        
        
        start_idx = randint(0, max_patch-min_patch)
        shrink_features = big_features[start_idx:start_idx+min_patch]

        if big_index==0:
            return (1 - lam) * shrink_features + lam * small_features  
        elif big_index==1:
            return (1 - lam) * small_features + lam * shrink_features 
    raise NotImplementedError  

def mixup_label(label_pair: Tuple[torch.Tensor, torch.Tensor],
        size_pair: Tuple[torch.Tensor, torch.Tensor],
        lam: float,
        mix_strategy,                  
        ) -> torch.Tensor:
    label1, label2 = label_pair      
    n_patch1, n_patch2 = size_pair[0], size_pair[1]
    total_patch = n_patch1 + n_patch2
    if mix_strategy=='sampling':        
        imbalanced_label = (n_patch1*label1+ n_patch2*label2)/total_patch # instance sampling        
        balanced_label = 0.5*label1+ 0.5*label2 # class ampling       
        return (1 - lam) * imbalanced_label + lam * balanced_label     
    else:
        return (1 - lam) * label1 + lam * label2
    

def mixup(features_pair: Tuple[torch.Tensor, torch.Tensor],
        label_pair: Tuple[torch.Tensor, torch.Tensor],
        lam: float,
        mix_strategy
        ) -> Tuple[torch.Tensor, torch.Tensor]:
   
    n_patch1, n_patch2 = features_pair[0].shape[0], features_pair[1].shape[0]        
    mixed_feats = mixup_features(features_pair, lam, mix_strategy)
    mixed_label = mixup_label(label_pair, (n_patch1, n_patch2), lam, mix_strategy)
    
    return mixed_feats, mixed_label

def cutmix_features(features_pair: Tuple[torch.Tensor, torch.Tensor],
        lam: float,
        mix_strategy,                  
        ) -> torch.Tensor:
    
    features1, features2 = features_pair 
    n_patch1, n_patch2 = features1.shape[0], features2.shape[0]
    feature_dim = features1.shape[1:]
       
    n_cutmix = int((1-lam)*n_patch2)
    
    
    if mix_strategy=='uniform':   
        # Directly sample the (1-lam) features from feature1 with replacement=True. 
        # The shuffle action has done in dataset's __getitem__ function, 
        # so that features2[:(n_patch2-n_cutmix) can get different features in the different epoch.       
        ids = torch.randint(0, n_patch1, (n_cutmix,))        
        return torch.cat([features1[ids], features2[:(n_patch2-n_cutmix)] ], dim=0)
    elif mix_strategy=='cutpaste':
        # the slide will be cut and paste to the other slide, 
        # and only sample features when the slide has not enough features.
        if n_patch1>n_cutmix:
            return torch.cat([features1[:n_cutmix], features2[:(n_patch2-n_cutmix)] ], dim=0)
        else:
           ids = torch.randint(0, n_patch1, (n_cutmix-n_patch1,)) 
           return torch.cat([features1, features1[ids], features2[:(n_patch2-n_cutmix)] ], dim=0)
    elif mix_strategy=='cutpaste2':
        start_idx2 = randint(0, n_patch2-n_cutmix)
        mix_feature = copy.deepcopy(features2)
        if n_patch1>n_cutmix:
            start_idx = randint(0, n_patch1-n_cutmix)  
            mix_feature[start_idx2:start_idx2+n_cutmix] = features1[start_idx:start_idx+n_cutmix]
            return mix_feature           
        else:
            multiplier = n_cutmix // n_patch1
            remainder = n_cutmix % n_patch1
            ids = list(range(n_patch1))*multiplier + list(range(remainder))  
            mix_feature[start_idx2:start_idx2+n_cutmix] = features1[ids]
            return mix_feature

    elif mix_strategy.startswith('rank'):        
        start_idx2 = randint(0, n_patch2-n_cutmix)
        mix_feature = copy.deepcopy(features2)
        start_idx = randint(0, n_patch1-n_cutmix)  
        mix_feature[start_idx2:start_idx2+n_cutmix] = features1[start_idx:start_idx+n_cutmix]
        return mix_feature

def cutmix_label(label_pair: Tuple[torch.Tensor, torch.Tensor],        
        size_pair: Tuple[torch.Tensor, torch.Tensor],
        lam: float,
        mix_strategy,                  
        ) -> torch.Tensor:
    label1, label2 = label_pair    
    return (1 - lam) * label1 + lam * label2
    

def cutmix(features_pair: Tuple[torch.Tensor, torch.Tensor],
        label_pair: Tuple[torch.Tensor, torch.Tensor],
        lam: float,
        mix_strategy
        ) -> Tuple[torch.Tensor, torch.Tensor]:    
    n_patch1, n_patch2 = features_pair[0].shape[0], features_pair[1].shape[0]           
    mixed_feats = cutmix_features(features_pair, lam, mix_strategy)
    mixed_label = cutmix_label(label_pair, (n_patch1, n_patch2), lam, mix_strategy)    
    return mixed_feats, mixed_label

def get_mix_strategy(args):
    # Define the set of wanted augmentations here.
    if args.mix_strategy.startswith("random") and args.mix_algorithm=='cutmix':
        strategy = choice(["uniform", "cutpaste"])
    elif args.mix_strategy.startswith("random") and args.mix_algorithm=='mixup':
        # mixup_strategy = choice(["direct", "shrink", "sampling", "duplicated", "duplicated2", "uniform", "uniform_partial"])
        if args.mix_strategy=="random_rank":
            strategy = choice(["rank", "shrink"])
        elif args.mix_strategy=="random_rank2":
            strategy = choice(["rank2", "shrink"])
        elif args.mix_strategy=="random_rank3":
            strategy = choice(["rank", "rank2", "shrink"])
        elif args.mix_strategy=="random_2rank":
            strategy = choice(["rank", "rank2"])
    else:
        strategy = args.mix_strategy
    return strategy


def mixup_by_rank(feats, label, zero_idx, milnet, strategy, mix, lam, top_k='min'):    
    milnet.eval()
    with torch.no_grad(): 
        if zero_idx is None:
            # Dsmil
            num_classes = 1 if label.view(2,-1).shape[-1]==1 else 2
            ins_predictions = []
            for i in range(len(feats)):                
                ins_prediction, _, _, _  = milnet(feats[i]) # (N,C)  
                ins_predictions.append(ins_prediction) # [(N1,C), (N2,C)]
            
            if strategy.startswith("rank2"):
                m_indices = []                
                for j in range(len(feats)):
                    if num_classes==1:
                        if label[j]==1:
                            _, indices = torch.sort(ins_predictions[j].squeeze(1), 0, descending=True) # (N,)
                        elif label[j]==0:
                            _, indices = torch.sort(ins_predictions[j].squeeze(1), 0, descending=False) # (N,)
                    elif num_classes==2:
                        if label[j, 1]==1:
                            _, indices = torch.sort(ins_predictions[j][:,1], 0, descending=True) # (N,)
                        elif label[j, 1]==0:
                            _, indices = torch.sort(ins_predictions[j][:,0], 0, descending=False) # (N,)                    
                    m_indices.append(indices) # [(N1,), (N2,)]
                
            elif strategy.startswith("rank"):
                m_indices = []
                for j in range(len(feats)):
                    if num_classes==1:                        
                            _, indices = torch.sort(ins_predictions[j].squeeze(1), 0, descending=True) # (N,)                
                    elif num_classes==2:
                        if label[j, 1]==1:
                           _, indices = torch.sort(ins_predictions[j][:,1], 0, descending=True) # (N,)
                        elif label[j, 1]==0:
                            _, indices = torch.sort(ins_predictions[j][:,0], 0, descending=True) # (N,)
                    m_indices.append(indices) # [(N1,), (N2,)]
        
        else:
            # Frmil
            _, ins_prediction = milnet(feats) # (2,N) 
            if strategy.startswith("rank2"):
                m_indices = []                
                num_classes = 1 if label.view(2,-1).shape[-1]==1 else 2
                for j in range(feats.shape[0]):
                    if (num_classes==2 and label[j,1]==1) or (num_classes==1 and label[j]==1):
                        _, indices = torch.sort(ins_prediction[[j]], 1, descending=True) # (1,N) 
                    elif (num_classes==2 and label[j,1]==0) or (num_classes==1 and label[j]==0):
                        _, indices = torch.sort(ins_prediction[[j]], 1, descending=False) # (1,N)    
                    else:
                        raise NotImplementedError
                    m_indices.append(indices)
                m_indices = torch.cat(m_indices, dim=0) # (2,N) 
            elif strategy.startswith("rank"):
                _, m_indices = torch.sort(ins_prediction, 1, descending=True) # (2,N) 
        
        # select features from ranking indices (m_indices)
        selected_indices = [] # (N,)
        if zero_idx is None:
            # Dsmil
            if strategy=="rank" or strategy=="rank2":
                min_patch = min(feats[0].shape[-2], feats[1].shape[-2])                
                selected_indices.append(m_indices[0][:min_patch])
                selected_indices.append(m_indices[1][:min_patch])
                        
            elif strategy.split('_')[-1].isdigit():
                ratio = float(strategy.split('_')[-1])*0.01
                min_patch = min(feats[0].shape[-2], feats[1].shape[-2])
                selected_indices.append(m_indices[0][:int(min_patch*ratio)])
                selected_indices.append(m_indices[1][:int(min_patch*ratio)]) 
            elif top_k!='min':
                selected_indices.append(m_indices[0][:int(top_k)])
                selected_indices.append(m_indices[1][:int(top_k)])
        else:    
            # Frmil 
            if strategy=="rank" or strategy=="rank2":
                selected_indices.append(m_indices[0][:min(zero_idx)])
                selected_indices.append(m_indices[1][:min(zero_idx)]) 
            elif strategy.split('_')[-1].isdigit():
                ratio = float(strategy.split('_')[-1])*0.01  
                selected_indices.append(m_indices[0][:(min(zero_idx)*ratio).long()]) 
                selected_indices.append(m_indices[1][:(min(zero_idx)*ratio).long()]) 
            elif top_k!='min':
                min_patch = min(top_k, min(zero_idx)) # avoid top-k bigger than feature number
                selected_indices.append(m_indices[0][:min_patch])
                selected_indices.append(m_indices[1][:min_patch])             
        
        # considering the order of patches  
        m_indices = [torch.sort(indices, 0)[1] for indices in selected_indices] # [(N,), (N,)] 
         
        selected_feats = [
            torch.index_select(feats[0], 0, m_indices[0]),
            torch.index_select(feats[1], 0, m_indices[1])
        ]

        if zero_idx is None:
            # Dsmil
            pass
        else: 
            # Frmil 
            selected_feats[0] = F.pad(selected_feats[0], (0, 0, 0, feats[0].shape[-2]-selected_feats[0].shape[-2]), "constant", 0)
            selected_feats[1] = F.pad(selected_feats[1], (0, 0, 0, feats[1].shape[-2]-selected_feats[1].shape[-2]), "constant", 0)
        
        mixed_feats, mixed_label = mix(
            (selected_feats[0], selected_feats[1]), 
            (label[0], label[1]), lam, strategy) 
      
    milnet.train()
    return mixed_feats, mixed_label

def train(train_combo_loader, milnet, pretrained_milnet, criterion, optimizer, args):
    milnet.train()
    kl_loss = nn.KLDivLoss(reduction="batchmean")    
    total_loss = 0    
    
    if args.mix_algorithm=='mixup':
        mix = mixup
    elif args.mix_algorithm=='cutmix':    
        mix = cutmix
    
    for i, (imbalanced_batch, balanced_batch) in enumerate(train_combo_loader):        
        optimizer.zero_grad()
        lam = np.random.beta(a=args.do_mixup, b=1)

        if args.mix_domain=='image':
            raise NotImplementedError("mix_domain=image is not implemented!!")            
            
        elif args.mix_domain=='feature':        
            imbalanced_label, imbalanced_feats = imbalanced_batch
            balanced_label, balanced_feats = balanced_batch                    

            imbalanced_label, imbalanced_feats = imbalanced_label.cuda(), imbalanced_feats.view(-1, args.feats_size).cuda()
            balanced_label, balanced_feats = balanced_label.cuda(), balanced_feats.view(-1, args.feats_size).cuda()
                        
            strategy = get_mix_strategy(args)
            if strategy.startswith("rank"):                 
                
                score_function = milnet
                mixed_feats, mixed_label = mixup_by_rank(
                    (imbalanced_feats, balanced_feats), 
                    torch.cat((imbalanced_label, balanced_label), dim=0), 
                    None , score_function, strategy, mix, lam, args.top_k
                ) 
            else:
                mixed_feats, mixed_label = mix((imbalanced_feats, balanced_feats), (imbalanced_label, balanced_label), lam, args.mix_strategy)
                        
            del balanced_feats
            del balanced_label
            del imbalanced_feats
            del imbalanced_label            
            
            mixed_feats = F.dropout(mixed_feats,p=0.20)            
            
            ins_prediction, bag_prediction, _, _ = milnet(mixed_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)
            bag_loss = criterion(bag_prediction.view(1, -1), mixed_label.view(1, -1))
            max_loss = criterion(max_prediction.view(1, -1), mixed_label.view(1, -1))            
                            
            loss = 0.5*bag_loss + 0.5*max_loss
            loss.backward()
            optimizer.step()
            total_loss = total_loss + loss.item() 
            sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_combo_loader), loss.item()))
        
    return total_loss / len(train_combo_loader)

def train_frmil(train_loader, milnet, pretrained_milnet, criterion, optimizer, args):
    milnet.train()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    if args.mix_algorithm=='mixup':        
        mix = mixup
    elif args.mix_algorithm=='cutmix':        
        mix = cutmix

    total_loss = 0
    if args.loss=='FrmilLoss':
        ce_weight  = [i for i in train_loader.dataset.count_dict.values()]
        ce_weight  = 1. / torch.tensor(ce_weight, dtype=torch.float)
        ce_weight  = ce_weight.cuda()
        bce_weight = train_loader.dataset.pos_weight.cuda()
    mag_loss = FeatMag(margin=args.mag).cuda()
    
    for i, (feats, label, _, zero_idx) in enumerate(train_loader):
        optimizer.zero_grad()
        
        norm_idx = torch.where(label == 0)[0].numpy()[0]
        ano_idx  = 1 - norm_idx

        label, feats = label.cuda().float(), feats.view(2, -1, args.feats_size).cuda() 
       
        lam = np.random.beta(a=args.do_mixup, b=args.do_mixup)
        
        strategy = get_mix_strategy(args)

        if strategy.startswith("rank"): 
            score_function = milnet        
            mixed_feats, mixed_label = mixup_by_rank(feats, label, zero_idx, score_function, strategy, mix, lam, args.top_k)
        else: 
            mixed_feats, mixed_label = mix(
                        (feats[0,:zero_idx[0]], feats[1,:zero_idx[1]]), 
                        (label[0], label[1]), lam, strategy)       
        
        n_pad = feats.shape[-2] - mixed_feats.shape[-2]        
        mixed_feats = F.pad(mixed_feats, (0, 0, 0, n_pad), "constant", 0)

        combined_features = torch.cat([feats, mixed_feats.unsqueeze(0)], dim=0) # (3,N,D)              
        combined_label = torch.cat([label, mixed_label.unsqueeze(0)],dim=0) # (3,1)        

        combined_features = F.dropout(combined_features,p=0.20)
              
        
        bag_prediction, query, ins_prediction = milnet(combined_features)  # (3,C), (3,N,D), (3,N)      
        max_prediction, _ = torch.max(ins_prediction, 1) # (3)
        
        if args.loss=='FrmilLoss':
            if args.num_classes==1:
                max_loss = F.binary_cross_entropy(max_prediction[[2]], combined_label[[2]].view(-1)) # (1), (1)                              
                bag_loss = criterion(bag_prediction[[2]], combined_label[[2]].view(1, -1)) # (1,1), (1,1)  
            elif args.num_classes==2:
                max_loss = F.binary_cross_entropy(max_prediction[[2]], combined_label[[2],1], weight=bce_weight) 
                bag_loss = F.cross_entropy(bag_prediction[[2]], combined_label[[2],1].long(), weight=ce_weight)            
            loss_ft  = mag_loss(query[ano_idx,:,:].unsqueeze(0),query[norm_idx,:,:].unsqueeze(0), w_scale=query.shape[1])
            loss = (bag_loss + max_loss + loss_ft) * (1./3)
        elif args.loss=='DsmilLoss':  
            # ablation study of loss_ft
            bag_loss = criterion(bag_prediction[2].view(1, -1), combined_label[2].view(1, -1))
            max_loss = criterion(max_prediction[2].view(1, -1), combined_label[2].view(1, -1))
            loss = 0.5*bag_loss + 0.5*max_loss        
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()   
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_loader.dataset), loss.item()))
    
    return total_loss / len(train_loader.dataset)


def dropout_patches(feats, p):
    idx = np.random.choice(np.arange(feats.shape[0]), int(feats.shape[0]*(1-p)), replace=False)
    sampled_feats = np.take(feats, idx, axis=0)
    pad_idx = np.random.choice(np.arange(sampled_feats.shape[0]), int(feats.shape[0]*p), replace=False)
    pad_feats = np.take(sampled_feats, pad_idx, axis=0)
    sampled_feats = np.concatenate((sampled_feats, pad_feats), axis=0)
    return sampled_feats

def test(test_loader, milnet, criterion, optimizer, args):
    milnet.eval()  
    total_loss = 0
    test_labels = []
    test_predictions = []
    Tensor = torch.cuda.FloatTensor
    with torch.no_grad():        
        for i, batch in enumerate(test_loader): 
            if args.model=='dsmil':
                label, feats = batch
                label, feats = label.cuda().float(), feats.view(-1, args.feats_size).cuda()            
                ins_prediction, bag_prediction, _, _ = milnet(feats)
                
            elif args.model=='frmil':
                feats, label, _  = batch
                feats, label = feats.cuda(), label.cuda().float()           
                bag_prediction, ins_prediction  = milnet(feats)
                ins_prediction = ins_prediction.squeeze(0)

            max_prediction, _ = torch.max(ins_prediction, 0)
            
            bag_loss = criterion(bag_prediction.view(1, -1), label.view(1, -1))            
            if args.num_classes!=1 and args.model=='frmil':   
                max_loss = criterion(max_prediction.view(1, -1), label[:, 1].view(1, -1))     
            else:
                max_loss = criterion(max_prediction.view(1, -1), label.view(1, -1))  
            loss = 0.5*bag_loss + 0.5*max_loss
            total_loss = total_loss + loss.item()
            sys.stdout.write('\r Testing bag [%d/%d] bag loss: %.4f' % (i, len(test_loader.dataset), loss.item()))
            test_labels.append([label.squeeze().cpu().numpy()])
            test_predictions.extend([(0.0*torch.sigmoid(max_prediction)+1.0*torch.sigmoid(bag_prediction)).squeeze().cpu().numpy()])
     
    test_labels = np.array(test_labels)    
    test_predictions = np.array(test_predictions)
      
    if len(test_labels.shape)==3:
        test_labels = test_labels.squeeze(1)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    prauc_value = multi_label_prauc(test_labels, test_predictions, args.num_classes)
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions<thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i]<thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_loader.dataset)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_loader.dataset)
    
    ## Cal avg score with threshold=0.5
    if args.num_classes==1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions>=0.5] = 1
        class_prediction_bag[test_predictions<0.5] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i]>=0.5] = 1
            class_prediction_bag[test_predictions[:, i]<0.5] = 0
            test_predictions[:, i] = class_prediction_bag
    bag_score = 0
    for i in range(0, len(test_loader.dataset)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_05_score = bag_score / len(test_loader.dataset)

    return total_loss / len(test_loader.dataset), avg_score, auc_value, thresholds_optimal, avg_05_score, prauc_value

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        c_auc = roc_auc_score(label, prediction)
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def multi_label_prauc(labels, predictions, num_classes):
    praucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(0, num_classes):
        label = labels[:, c]
        prediction = predictions[:, c]
        precision, recall, thresholds = precision_recall_curve(label, prediction)        
        auc_precision_recall = auc(recall, precision)   
        praucs.append(auc_precision_recall)        
    return praucs

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def main():
    parser = argparse.ArgumentParser(description='Train DSMIL on 20x patch features learned by SimCLR')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--feats_size', default=512, type=int, help='Dimension of the feature size [512]')
    parser.add_argument('--n_heads', default=8, type=int, help='the number of heads for self-attention')
    parser.add_argument('--sampling', type=str, default='instance', help='sampling mode (instance, class, sqrt, prog)')
    parser.add_argument('--do_mixup', type=float, default=0.1, help='mixup coeff (so far only for multi-class')    
    parser.add_argument('--mix_algorithm', type=str, default='mixup', help='do mixup or cutmix')
    parser.add_argument('--mix_domain', type=str, default='feature', help='do mixup in feature/image domain')
    parser.add_argument('--patch_batch_size', type=int, default=512, help='the number of patches for 1 batch (image domain)')
    parser.add_argument('--mix_strategy', type=str, default='rank', help='mixup two slides by different strategy')
    parser.add_argument('--top_k', type=str, default='min', help='the number of selected feature number')    
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate [0.0002]')
    parser.add_argument('--num_epochs', default=200, type=int, help='Number of total training epochs [40|200]')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight decay [5e-3]')
    parser.add_argument('--dataset', default='TCGA-lung-default', type=str, help='Dataset folder name')
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Dataset directory')
    parser.add_argument('--split', default=0.2, type=float, help='Training/Validation split [0.2]')
    parser.add_argument('--model', default='dsmil', type=str, help='MIL model [dsmil]')
    parser.add_argument('--loss', default='DsmilLoss', type=str, help='loss for FRMIL')
    parser.add_argument("--mag", type=float, default=8.48, help='margin used in the feature loss (cm16)')
    parser.add_argument('--dropout_patch', default=0, type=float, help='Patch dropout rate [0]')
    parser.add_argument('--dropout_node', default=0, type=float, help='Bag classifier dropout rate [0]')
    parser.add_argument('--dropout_input', default=0.2, type=float, help='The ratio of dropout')
    parser.add_argument('--non_linearity', default=1, type=float, help='Additional nonlinear operation [0]')
    parser.add_argument('--num_workers', default=4, type=int, help='num_workers for dataloader')    
    parser.add_argument('--save_all', action='store_true', help='Save all models for every epoch') 
    parser.add_argument('--pretrain', action='store_true', help='Whether load pretrain model or not') 
    parser.add_argument('--fix_score_function', type=str, default=None, help='The different approaches for training score function.') 
    parser.add_argument('--distill_loss', type=str, default='ScoreSigmoidKL', help='The loss function of distillation.') 
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    if args.model == 'dsmil':
        import dsmil as mil
    elif args.model == 'abmil':
        import abmil as mil
    elif args.model == 'frmil':
        from frmil import FRMIL
        milnet = FRMIL(args).cuda()
        print("Using FRMIL model!!")
    
    if args.mix_domain=='feature' and args.model == 'dsmil':    
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    
    milnet_test =  milnet

    if args.top_k != 'min':
        save_path = f"{args.dataset}_{args.model}_k_{args.top_k}_{args.mix_algorithm}_{args.mix_domain}"
        print("Testing differemt top_k number:", args.top_k)
    else:
        save_path = f"{args.dataset}_{args.model}_{args.loss}_{args.mix_algorithm}_{args.mix_domain}"

    # Load pretrained model
    if args.pretrain:
        milnet = load_model(milnet, args.dataset, args.model, loss=args.loss)
        save_path = save_path + "_pretrain"


    pretrained_milnet = None # 'finetune' 
    save_path = os.path.join(  
        'weights',          
        f"{save_path}_{args.mix_strategy}", 
        datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
    )
    
        
    os.makedirs(save_path, exist_ok=True)
    log_filename = os.path.join(save_path, 'log.txt')    
    print(f"log_filename: {log_filename}")    
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
    
    if args.dataset.startswith("C16_dataset"):
        bags_csv = os.path.join('datasets', args.dataset, 'training', 'training.csv')    
    else:
        bags_csv = os.path.join('datasets', args.dataset, args.dataset+'.csv')
        print("Loading bags_csv:", bags_csv)    
        bags_path = pd.read_csv(bags_csv)
    
    if args.dataset.startswith("C16_dataset"):        
        train_path = pd.read_csv(os.path.join('datasets', args.dataset, 'training', 'training.csv'))
        test_path = pd.read_csv(os.path.join('datasets', args.dataset, 'testing', 'testing.csv'))
        reference_csv = pd.read_csv(
            os.path.join('..', args.dataset_dir, 'C16_dataset', 'testing', 'reference.csv'),
            names=['slide_name', 'label'],
            usecols=[0, 1],
            index_col=False
        )        
        
        reference_csv['label'] = reference_csv['label'].replace({'Tumor':1, 'Normal':0})       
        test_path = test_path.sort_values(by=['0'], ignore_index=True)
        test_path['label'] = reference_csv['label']

        print(f"Loading training set: {os.path.join('datasets', args.dataset, 'training', 'training.csv')}")  
        print(f"Loading testing set: {os.path.join('datasets', args.dataset, 'testing', 'testing.csv')}")         
        train_class_count = np.unique(train_path.iloc[:,1], return_counts=True)[1]       
        print(f"# of positive samples in training set: {train_class_count[1]}/{train_path.shape[0]}")         
        test_class_count = np.unique(test_path.iloc[:,1], return_counts=True)[1]        
        print(f"# of positive samples in testing set: {test_class_count[1]}/{test_path.shape[0]}") 
    else:    
        train_path = bags_path.iloc[0:int(len(bags_path)*(1-args.split)), :]
        test_path = bags_path.iloc[int(len(bags_path)*(1-args.split)):, :] 
    
    if args.model=="dsmil":
        train_dataset = ClassDataset(train_path, args)
        test_dataset = ClassDataset(test_path, args)            
        val_dataset = train_dataset
    elif args.model=="frmil":            
        train_dataset = FrmilClassDataset(df=train_path, mode='train', num_classes=args.num_classes, batch=True)            
        test_dataset  = FrmilClassDataset(df=test_path, mode='test', num_classes=args.num_classes)
        val_dataset   = FrmilClassDataset(df=train_path, mode='train', num_classes=args.num_classes) 
        train_sampler = CategoriesSampler(train_dataset.labels, n_batch=len(train_dataset), n_cls=2, n_per=1)  

    if args.model=="dsmil":
        train_loader = DataLoader(dataset=train_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), shuffle=True, drop_last=False)
    elif args.model=="frmil":
        train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)
    
    
    if args.model != "frmil":        
        train_loader = get_combo_loader(train_loader, base_sampling=args.sampling)

    print("milnet:")
    print(milnet)

    best_val_score = 0     
    best_val_loss_bag = 0
    best_val_avg_score = 0   
    run = len(glob.glob(os.path.join(save_path, '*.pth')))   
 
    if args.pretrain:            
        origin_loss_bag, origin_avg_score, origin_aucs, origin_thresholds_optimal, origin_avg_05_score, origin_praucs = test(test_loader, milnet, criterion, optimizer, args)
        print_result(origin_avg_score, origin_avg_05_score, origin_aucs, origin_praucs, result_type='Pretrain', dataset=args.dataset)

    for epoch in range(1, args.num_epochs):    
        start = time.time()        
        if args.model == 'frmil':
            train_loss_bag = train_frmil(train_loader, milnet, pretrained_milnet, criterion, optimizer, args) # iterate all bags            
        else:
            train_loss_bag = train(train_loader, milnet, pretrained_milnet, criterion, optimizer, args) # iterate all bags
       

        val_loss_bag, val_avg_score, val_aucs, val_thresholds_optimal, val_avg_05_score, val_praucs = test(val_loader, milnet_test, criterion, optimizer, args)
        test_loss_bag, avg_score, aucs, thresholds_optimal, avg_05_score, praucs = test(test_loader, milnet_test, criterion, optimizer, args)
        if args.dataset=='Lung':
            output_string = (
                '\r Epoch [%d/%d] time: %.4f train loss: %.4f, val loss: %.4f, val acc: %.4f, test loss: %.4f, test acc: %.4f, test 0.5 acc: %.4f, val_auc_LUAD: %.4f, val_auc_LUSC: %.4f, auc_LUAD: %.4f, auc_LUSC: %.4f, val_prauc_LUAD: %.4f, val_prauc_LUSC: %.4f, prauc_LUAD: %.4f, prauc_LUSC: %.4f' %
                (epoch, args.num_epochs, time.time()-start, train_loss_bag, val_loss_bag, val_avg_score, test_loss_bag, avg_score, avg_05_score, val_aucs[0], val_aucs[1], aucs[0], aucs[1], val_praucs[0], val_praucs[1], praucs[0], praucs[1])
            )            
        else: 
            output_string = (
                '\r Epoch [%d/%d] time: %.4f train loss: %.4f, val loss: %.4f, val_acc: %.4f, test loss: %.4f, test acc: %.4f, test 0.5 acc: %.4f, val_AUC: ' % 
                (epoch, args.num_epochs, time.time()-start, train_loss_bag, val_loss_bag, val_avg_score, test_loss_bag, avg_score, avg_05_score) 
                + '|'.join('class-{}>>{:.4f}'.format(*k) for k in enumerate(val_aucs))
                + ', AUC: '
                + '|'.join('class-{}>>{:.4f}'.format(*k) for k in enumerate(aucs))
                + ', val_PRAUC: '
                + '|'.join('class-{}>>{:.4f}'.format(*k) for k in enumerate(val_praucs))
                + ', PRAUC: '
                + '|'.join('class-{}>>{:.4f}'.format(*k) for k in enumerate(praucs))
            )

        print(output_string)
        with open(log_filename, "a") as log_file:
            log_file.write(output_string + '\n')

        scheduler.step()        
        current_score = (sum(aucs) + avg_score + sum(praucs))/3        
        current_val_score = (sum(val_aucs) + val_avg_score + sum(val_praucs))/3
        if args.save_all or epoch==args.num_epochs-1:
            torch.save(milnet.state_dict(), os.path.join(save_path, f"checkpoint_{epoch}.pth"))        
        torch.save(milnet.state_dict(), os.path.join(save_path, f"temp.pth"))

        if current_val_score > best_val_score or (current_val_score==best_val_score and val_loss_bag<best_val_loss_bag): 
            best_val_score = current_val_score             
            best_val_loss_bag = val_loss_bag

            best_val_aucs = aucs
            best_val_praucs = praucs
            best_val_avg_score = avg_score 
            best_val_avg_05_score = avg_05_score 
             
            save_name = os.path.join(save_path, 'best.pth')
            torch.save(milnet.state_dict(), save_name)
            
            
            if args.dataset=='Lung':
                print('Best model saved at: ' + save_name + ' Best thresholds: LUAD %.4f, LUSC %.4f' % (thresholds_optimal[0], thresholds_optimal[1]))
                with open(log_filename, "a") as log_file:
                    log_file.write('Best model saved at: ' + save_name + ' Best thresholds: LUAD %.4f, LUSC %.4f\n' % (thresholds_optimal[0], thresholds_optimal[1]))
            else:
                print('Best model saved at: ' + save_name)
                print('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)))
                with open(log_filename, "a") as log_file:
                    log_file.write('Best model saved at: ' + save_name)
                    log_file.write('Best thresholds ===>>> '+ '|'.join('class-{}>>{}'.format(*k) for k in enumerate(thresholds_optimal)) + '\n')
    
    # Final Result
    if args.dataset=='Lung':        
        result_string1 = f"The Val Model: ACC {best_val_avg_score:.4f} | ACC05 {best_val_avg_05_score:.4f} |auc_LUAD: {best_val_aucs[0]:.4f}/{best_val_praucs[0]:.4f}, auc_LUSC: {best_val_aucs[1]:.4f}/{best_val_praucs[1]:.4f}"
        result_string2 = f"The Last Model: ACC {avg_score:.4f} | ACC05 {avg_05_score:.4f} |auc_LUAD: {aucs[0]:.4f}/{praucs[0]:.4f}, auc_LUSC: {aucs[1]:.4f}/{praucs[1]:.4f}"
        
    else:               
        result_string1 = f"The Val Model: ACC {best_val_avg_score:.4f} | ACC05 {best_val_avg_05_score:.4f} | AUC {best_val_aucs[0]:.4f}/{best_val_praucs[0]:.4f}"
        result_string2 = f"The Last Model: ACC {avg_score:.4f} | ACC05 {avg_05_score:.4f} | AUC {aucs[0]:.4f}/{praucs[0]:.4f}"
        
    print(result_string1)
    print(result_string2)
    with open(log_filename, "a") as log_file:
        log_file.write(result_string1 + '\n')
        log_file.write(result_string2 + '\n')
    
    if args.top_k != 'min':
        csv_path = os.path.join('weights', f'results_{args.dataset}_topk.csv')
    else:
        csv_path = os.path.join('weights', f'results_{args.dataset}_{args.num_classes}class.csv')
    
    if os.path.exists(csv_path):
        fp = open(csv_path, 'a')
    elif args.dataset=='Lung':
        fp = open(csv_path, 'w')
        fp.write('method,epoch,fix_score_function,dropout,val_acc,acc,val_acc05,acc05,val_auc_LUAD,auc_LUAD,val_auc_LUSC,auc_LUSC,threshold_LUAD,threshold_LUSC,val_prauc_LUAD,prauc_LUAD,val_prauc_LUSC,prauc_LUSC\n')
    elif args.num_classes==1:
        fp = open(csv_path, 'w')        
        fp.write('method,epoch,fix_score_function,dropout,val_acc,acc,val_acc05,acc05,val_auc,auc,threshold,val_prauc,prauc\n')
    elif args.num_classes==2:
        fp = open(csv_path, 'w')        
        fp.write('method,epoch,fix_score_function,dropout,val_acc,acc,val_acc05,acc05,val_auc[0],auc[0],val_auc[1],auc[1],threshold[0],threshold[1],val_prauc[0],prauc[0],val_prauc[1],prauc[1]\n')

    method_name = save_path.split('/')[1]
    print(f"Saving {method_name}....")    
    
    fix_score_function = args.fix_score_function if args.fix_score_function else '-'
    if args.pretrain:
        if args.num_classes==1:
            fp.write(f'{method_name}_origin,x,x,-,'
            f'{origin_avg_score:.4f},x,'    
            f'{origin_avg_05_score:.4f},x,'     
            f'{origin_aucs[0]:.4f},x,'        
            f'{origin_thresholds_optimal[0]:.4f},'
            f'{origin_praucs[0]:.4f},x\n')
        elif args.num_classes==2:
            fp.write(f'{method_name}_origin,x,x,-,'
            f'{origin_avg_score:.4f},x,'  
            f'{origin_avg_05_score:.4f},x,'       
            f'{origin_aucs[0]:.4f},x,'
            f'{origin_aucs[1]:.4f},x,'
            f'{origin_thresholds_optimal[0]:.4f},{origin_thresholds_optimal[1]:.4f},'
            f'{origin_praucs[0]:.4f},x,'
            f'{origin_praucs[1]:.4f},x\n')

    if args.num_classes==1:
        fp.write(f'{method_name},{epoch},{fix_score_function},{args.dropout_input},'
        f'{best_val_avg_score:.4f},{avg_score:.4f},'    
        f'{best_val_avg_05_score:.4f},{avg_05_score:.4f},'     
        f'{best_val_aucs[0]:.4f},{aucs[0]:.4f},'        
        f'{thresholds_optimal[0]:.4f},'
        f'{best_val_praucs[0]:.4f},{praucs[0]:.4f}\n')
    elif args.num_classes==2:
        fp.write(f'{method_name},{epoch},{fix_score_function},{args.dropout_input},'
        f'{best_val_avg_score:.4f},{avg_score:.4f},'  
        f'{best_val_avg_05_score:.4f},{avg_05_score:.4f},'       
        f'{best_val_aucs[0]:.4f},{aucs[0]:.4f},'
        f'{best_val_aucs[1]:.4f},{aucs[1]:.4f},'
        f'{thresholds_optimal[0]:.4f},{thresholds_optimal[1]:.4f},'
        f'{best_val_praucs[0]:.4f},{praucs[0]:.4f},'
        f'{best_val_praucs[1]:.4f},{praucs[1]:.4f}\n')

    fp.close()
    print()

if __name__ == '__main__':
    main()