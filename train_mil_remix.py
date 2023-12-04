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
from scipy.spatial.distance import cdist

from dataset import get_sampling_probabilities
from dataset import ClassDatasetRemix
from dataset_frmil import RemixToFrmilClassDataset
from train_mil import FeatMag
from samplers import CategoriesSampler
from utils import load_model, print_result

def get_bag_feats_v2(feats, bag_label, args):
    if isinstance(feats, str):
        # if feats is a path, load it
        feats = feats.split(',')[0]
        feats = torch.Tensor(np.load(feats)).cuda()
    
    feats = feats[np.random.permutation(len(feats))]
    if args.num_classes != 1:
        # mannual one-hot encoding, following dsmil
        label = np.zeros(args.num_classes)
        if int(bag_label) <= (len(label) - 1):
            label[int(bag_label)] = 1
        bag_label = Variable(torch.FloatTensor([label]).cuda())
        
    return bag_label, feats


def convert_label(labels, num_classes=2):
    # one-hot encoding for multi-class labels
    if num_classes > 1:
        # one-hot encoding
        converted_labels = np.zeros((len(labels), num_classes))
        for ix in range(len(labels)):
            converted_labels[ix, int(labels[ix])] = 1
        return converted_labels
    else:
        # return binary labels
        return labels


def inverse_convert_label(labels):
    # one-hot decoding
    if len(np.shape(labels)) == 1:
        return labels
    else:
        converted_labels = np.zeros(len(labels))
        for ix in range(len(labels)):
            converted_labels[ix] = np.argmax(labels[ix])
        return converted_labels


def mix_aug(src_feats, tgt_feats, mode='replace', rate=0.3, strength=0.5, shift=None):
    assert mode in ['replace', 'append', 'interpolate', 'cov', 'joint']
    auged_feats = [_ for _ in src_feats.reshape(-1, 512)]
    tgt_feats = tgt_feats.reshape(-1, 512)
    closest_idxs = np.argmin(cdist(src_feats.reshape(-1, 512), tgt_feats), axis=1)
    if mode != 'joint':
        for ix in range(len(src_feats)):
            if np.random.rand() <= rate:
                if mode == 'replace':
                    auged_feats[ix] = tgt_feats[closest_idxs[ix]]
                elif mode == 'append':
                    auged_feats.append(tgt_feats[closest_idxs[ix]])
                elif mode == 'interpolate':
                    generated = (1 - strength) * auged_feats[ix] + strength * tgt_feats[closest_idxs[ix]]
                    auged_feats.append(generated)
                elif mode == 'cov':
                    generated = auged_feats[ix][np.newaxis, :] + strength * shift[closest_idxs[ix]][np.random.choice(200, 1)]
                    auged_feats.append(generated.flatten())
                else:
                    raise NotImplementedError
    else:
        for ix in range(len(src_feats)):
            if np.random.rand() <= rate:
                # replace
                auged_feats[ix] = tgt_feats[closest_idxs[ix]]
            if np.random.rand() <= rate:
                # append
                auged_feats.append(tgt_feats[closest_idxs[ix]])
            if np.random.rand() <= rate:
                # interpolate
                generated = (1 - strength) * auged_feats[ix] + strength * tgt_feats[closest_idxs[ix]]
                auged_feats.append(generated)
            if np.random.rand() <= rate:
                # covary
                generated = auged_feats[ix][np.newaxis, :] + strength * shift[closest_idxs[ix]][np.random.choice(200, 1)]
                auged_feats.append(generated.flatten())
    return np.array(auged_feats)


def mix_the_bag_aug(bag_feats, idx, train_feats, train_labels, args, semantic_shifts=None):
    if args.mode is not None:
        # randomly select one bag from the same class
        positive_idxs = np.argwhere(train_labels.cpu().numpy() == train_labels[idx].item()).reshape(-1)
        selected_id = np.random.choice(positive_idxs)
        # lambda parameter
        strength = np.random.uniform(0, 1)
        bag_feats = mix_aug(bag_feats.cpu().numpy(), train_feats[selected_id].cpu().numpy(),
                            shift=semantic_shifts[selected_id] if args.mode == 'joint' or args.mode == 'cov' else None,
                            rate=args.rate, strength=strength, mode=args.mode)
        bag_feats = torch.Tensor([bag_feats]).cuda()
    bag_feats = bag_feats.view(-1, args.feats_size)
    return bag_feats

def train_dsmil(train_loader, semantic_shifts, milnet, pretrained_milnet, criterion, optimizer, args):
    milnet.train()
    # csvs = shuffle(train_df).reset_index(drop=True)
    total_loss = 0
    # bc = 0
    Tensor = torch.cuda.FloatTensor
    # for i in range(len(train_df)):   
    for i, batch in enumerate(train_loader):
        optimizer.zero_grad()
        # if i >3 :
        #     break        
        label, feats = batch        
        label, feats = label.cuda(), feats.view(-1, args.feats_size).cuda()

        if torch.isnan(feats).sum() > 0:
            continue
        
        feats = mix_the_bag_aug(feats, i, train_loader.dataset.feats, train_loader.dataset.labels, args, semantic_shifts)        
        feats = F.dropout(feats,p=args.dropout_input)
        ins_prediction, bag_prediction, _, _ = milnet(feats)
        # bag_loss = criterion(bag_prediction.view(1, -1), bag_label.view(1, -1))
        # max_loss = criterion(max_prediction.view(1, -1), bag_label.view(1, -1))
        max_prediction, _ = torch.max(ins_prediction, 0)
        bag_loss = criterion(bag_prediction.view(1, -1), label.view(1, -1))
        max_loss = criterion(max_prediction.view(1, -1), label.view(1, -1))
        loss = 0.5*bag_loss + 0.5*max_loss
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        # sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_loader.dataset), loss.item()))
    # return total_loss / len(train_df)
    return total_loss / len(train_loader.dataset)

def train_frmil(train_loader, semantic_shifts, milnet, pretrained_milnet, criterion, optimizer, args):
    milnet.train()

    total_loss = 0    
    ce_weight  = [i for i in train_loader.dataset.count_dict.values()]
    ce_weight  = 1. / torch.tensor(ce_weight, dtype=torch.float)
    ce_weight  = ce_weight.cuda()
    bce_weight = train_loader.dataset.pos_weight.cuda()
    mag_loss   = FeatMag(margin=args.mag).cuda()
    # for i in range(len(train_df)):   
    # for i, batch in enumerate(train_loader):
    for i, (feats, label, _, zero_idx) in enumerate(train_loader):
        optimizer.zero_grad()
        # if i >20 :
        #     break
        
        # label, feats = batch # (2, 1), (2,N,D) 
        norm_idx = torch.where(label == 0)[0].numpy()[0]
        ano_idx  = 1 - norm_idx

        label, feats = label.cuda().float(), feats.view(2, -1, args.feats_size).cuda()         
        if torch.isnan(feats).sum() > 0:
            continue
        
        feat1, feat2 = feats[0], feats[1]
        feat1 = mix_the_bag_aug(feat1, i, torch.Tensor(train_loader.dataset.feats), torch.Tensor(train_loader.dataset.labels), args, semantic_shifts)
        feat2 = mix_the_bag_aug(feat2, i, torch.Tensor(train_loader.dataset.feats), torch.Tensor(train_loader.dataset.labels), args, semantic_shifts)
        n_patch1 = feat1.shape[0]  
        n_patch2 = feat2.shape[0]
        if n_patch1>n_patch2:
            n_pad = n_patch1 - n_patch2       
            feat2 = F.pad(feat2, (0, 0, 0, n_pad), "constant", 0)            
        else: 
            n_pad = n_patch2 - n_patch1       
            feat1 = F.pad(feat1, (0, 0, 0, n_pad), "constant", 0)            
        
        feats = torch.stack([feat1, feat2], axis=0)
                
        feats = F.dropout(feats,p=0.20)
        bag_prediction, query, ins_prediction = milnet(feats)  # (2,C), (2,N,D), (2,N) 
        max_prediction, _ = torch.max(ins_prediction, 1) # (2)
        
        if args.num_classes==1:
            max_loss = F.binary_cross_entropy(max_prediction, label.squeeze()) # (2), (2)
            # bag_loss = F.cross_entropy(bag_prediction, label)
            bag_loss = criterion(bag_prediction, label.view(2, -1)) 
        elif args.num_classes==2:                
            max_loss = F.binary_cross_entropy(max_prediction, label[:, 1], weight=bce_weight)
            bag_loss = F.cross_entropy(bag_prediction, label[:, 1].long(), weight=ce_weight)            
        if args.dataset.startswith("Lung") or args.dataset.startswith("C16") or args.dataset.startswith("imb_c16"):
            ano_idx, norm_idx = norm_idx, ano_idx                
        loss_ft  = mag_loss(query[ano_idx,:,:].unsqueeze(0),query[norm_idx,:,:].unsqueeze(0), w_scale=query.shape[1])
        loss = (bag_loss + max_loss + loss_ft) * (1./3)
        
        
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()
        # sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_df), loss.item()))        
        sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_loader.dataset), loss.item()))
    # return total_loss / len(train_df)
    return total_loss / len(train_loader.dataset)

def train(train_combo_loader, milnet, pretrained_milnet, criterion, optimizer, args):
    milnet.train()
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    # kl_loss = nn.KLDivLoss()
    # csvs = shuffle(train_df).reset_index(drop=True)
    total_loss = 0    
    # bc = 0    
    # Tensor = torch.cuda.FloatTensor
 
    
    for i, (imbalanced_batch, balanced_batch) in enumerate(train_combo_loader):
        # if i>2:
        #     break 
        optimizer.zero_grad()
        lam = np.random.beta(a=args.do_mixup, b=1)               
        imbalanced_label, imbalanced_feats = imbalanced_batch
        balanced_label, balanced_feats = balanced_batch                    

        imbalanced_label, imbalanced_feats = imbalanced_label.cuda(), imbalanced_feats.view(-1, args.feats_size).cuda()
        balanced_label, balanced_feats = balanced_label.cuda(), balanced_feats.view(-1, args.feats_size).cuda()
        
        strategy = get_mix_strategy(args)
        if strategy.startswith("rank"):            
            if args.fix_score_function=='fixed':
                score_function = pretrained_milnet
            else:
                score_function = milnet
            
            mixed_feats, mixed_label = mixup_by_rank(
                (imbalanced_feats, balanced_feats), 
                torch.cat((imbalanced_label, balanced_label), dim=0), 
                None , score_function, strategy, mix, lam
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
        if args.fix_score_function=='distilled':
            sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f | distill loss: %.8f' % (i, len(train_combo_loader), loss.item(), distill_loss.item()))
        else:
            sys.stdout.write('\r Training bag [%d/%d] bag loss: %.4f' % (i, len(train_combo_loader), loss.item()))
        
    return total_loss / len(train_combo_loader)


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
    parser.add_argument('--mix_strategy', type=str, default='sampling', help='mixup two slides by sampling strategy')    
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
    parser.add_argument('--save_all', action='store_true', help='save all models for every epoch') 
    parser.add_argument('--pretrain', action='store_true', help='Whether load pretrain model or not') 
    parser.add_argument('--fix_score_function', type=str, default=None, help='The different approaches for training score function.') 
    parser.add_argument('--distill_loss', type=str, default='ScoreSigmoidKL', help='The loss function of distillation.') 
    # ReMix Parameters
    parser.add_argument('--num_prototypes', default=None, type=int, help='Number of prototypes per bag')
    parser.add_argument('--mode', default=None, type=str,
                        choices=['None', 'replace', 'append', 'interpolate', 'cov', 'joint'],
                        help='Augmentation method')
    parser.add_argument('--rate', default=0.5, type=float, help='Augmentation rate')    
    
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    
    if args.dataset.startswith("Lung") or args.dataset.startswith("C16") or args.dataset.startswith("imb_c16"):
        print("the mag of neg is larger than the mag of pos!!")


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
    
    elif args.mix_domain=='image' and args.model == 'dsmil':
        if args.dataset.startswith("C16_dataset"):
            if 'single' in args.dataset:
                embedder_path = os.path.join('embedder', args.dataset, 'training', 'embedder.pth')
            else:
                embedder_path = os.path.join('embedder', args.dataset, 'training', 'embedder-low.pth')
        else:
            if 'single' in args.dataset:
                embedder_path = os.path.join('embedder', args.dataset, 'embedder.pth')
            else:
                embedder_path = os.path.join('embedder', args.dataset, 'embedder-low.pth')
        print("Loading ResNet18 pretrained by SimCLR:", embedder_path)
        resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
        for param in resnet.parameters():
            param.requires_grad = False
        resnet.fc = nn.Identity()
        i_classifier = mil.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()        
        state_dict_weights = torch.load(embedder_path)
        new_state_dict = OrderedDict()
        for i in range(4):
            state_dict_weights.popitem()
        state_dict_init = i_classifier.state_dict()
        for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
            name = k_0
            new_state_dict[name] = v

        i_classifier.load_state_dict(new_state_dict, strict=False)
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, dropout_v=args.dropout_node, nonlinear=args.non_linearity).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()
    
    if args.mix_domain=='feature':
       milnet_test =  milnet
    elif args.mix_domain=='image' and args.model == 'dsmil':
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda() 
        milnet_test = mil.MILNet(i_classifier, b_classifier).cuda()

    save_path = f"{args.dataset}_{args.model}_{args.loss}_remix_{args.num_prototypes}_{args.mode}_{args.rate}"
    
    # Load pretrained model
    if args.pretrain:
        milnet = load_model(milnet, args.dataset, args.model, loss=args.loss)
        save_path = save_path + "_pretrain"

    pretrained_milnet = None # 'finetune'    
    save_path = os.path.join(  
        'weights',          
        f"{save_path}", 
        datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
    )
        
    os.makedirs(save_path, exist_ok=True)
    log_filename = os.path.join(save_path, 'log.txt')    
    print(f"log_filename: {log_filename}")    
    
    criterion = nn.BCEWithLogitsLoss()
    
    optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 0.000005)
 
    if args.dataset is not None:
        train_labels_pth = f'../ReMix/datasets/{args.dataset}/remix_processed/train_bag_labels.npy'
        test_labels_pth = f'../ReMix/datasets/{args.dataset}/remix_processed/test_bag_labels.npy'

        # loading the list of test data
        test_feats_path = open(f'../ReMix/datasets/{args.dataset}/remix_processed/test_list.txt', 'r').readlines()
                
        if args.num_prototypes is not None:
            # load reduced-bag
            train_feats_pth = f'../ReMix/datasets/{args.dataset}/remix_processed/train_bag_feats_proto_{args.num_prototypes}.npy'
            # loading features
            train_feats = np.load(train_feats_pth, allow_pickle=True)
            # train_feats = torch.Tensor(train_feats).cuda()

            if args.mode == 'cov' or args.mode == 'joint':
                # loading semantic shift vectors
                train_shift_bank_pth = f'../ReMix/datasets/{args.dataset}/remix_processed/train_bag_feats_shift_{args.num_prototypes}.npy'
                semantic_shifts = np.load(f'{train_shift_bank_pth}')
            else:
                semantic_shifts = None
        else:
            # when train_feats is None, loading them directly from the dataset npy folder.
            train_feats = open(f'../ReMix/datasets/{args.dataset}/remix_processed/train_list.txt', 'r').readlines()
            train_feats = np.array(train_feats)
            semantic_shifts = None
            
        # loading labels
        train_labels, test_labels = np.load(train_labels_pth), np.load(test_labels_pth)        
        print("train_feats:", train_feats.shape) 
             
        train_class_count = np.unique(train_labels, return_counts=True)[1]        
        print(f"# of positive samples in training set: {train_class_count[1]}/{len(train_labels)}")         
        test_class_count = np.unique(test_labels, return_counts=True)[1]        
        print(f"# of positive samples in testing set: {test_class_count[1]}/{len(test_labels)}") 

    if args.model=="dsmil":
        train_dataset = ClassDatasetRemix(train_feats, train_labels, args)
        test_dataset = ClassDatasetRemix(test_feats_path, test_labels, args, mode='test')        
        val_dataset = train_dataset
    elif args.model=="frmil":            
        train_dataset = RemixToFrmilClassDataset(train_feats, train_labels, mode='train', num_classes=args.num_classes, batch=True)
        test_dataset  = RemixToFrmilClassDataset(test_feats_path, test_labels, mode='test', num_classes=args.num_classes)
        val_dataset   = RemixToFrmilClassDataset(train_feats, train_labels, mode='train', num_classes=args.num_classes)
        train_sampler = CategoriesSampler(train_dataset.labels, n_batch=len(train_dataset), n_cls=2, n_per=1)  

    if args.model=="dsmil":
        train_loader = DataLoader(dataset=train_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), shuffle=True, drop_last=False)
    elif args.model=="frmil":
        train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), drop_last=False)
    
    print("milnet:")
    print(milnet)

    best_val_score = 0     
    best_val_loss_bag = 0
    best_val_avg_score = 0   
    run = len(glob.glob(os.path.join(save_path, '*.pth')))

    if args.pretrain:            
        origin_loss_bag, origin_avg_score, origin_aucs, origin_thresholds_optimal, origin_avg_05_score, origin_praucs = test(test_loader, milnet, criterion, optimizer, args)
        print_result(origin_avg_score, origin_avg_05_score, origin_aucs, result_type='Pretrain', dataset=args.dataset)
   
    for epoch in range(1, args.num_epochs):
        start = time.time()        
        if args.model == 'frmil':
            train_loss_bag = train_frmil(train_loader, semantic_shifts, milnet, pretrained_milnet, criterion, optimizer, args) # iterate all bags            
        else:
            train_loss_bag = train_dsmil(train_loader, semantic_shifts, milnet, pretrained_milnet, criterion, optimizer, args) # iterate all bags
        
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
        
    csv_path = os.path.join('weights', f'results_{args.dataset}_remix_{args.num_classes}class.csv')
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