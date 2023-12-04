import dsmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob, copy
import pandas as pd
import numpy as np
from PIL import Image
from collections import OrderedDict
from sklearn.utils import shuffle
from tqdm import tqdm


class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform
    def __len__(self):
        return len(self.files_list)
    def __getitem__(self, idx):
        temp_path = self.files_list[idx]
        img = os.path.join(temp_path)
        img = Image.open(img)
        sample = {'input': img}
        
        if self.transform:
            sample = self.transform(sample)
        return sample 

class HighBagDataset():
    def __init__(self, csv_file, low_feats_list, transform=None):
        self.low_patch_files = csv_file
        self.low_feats_list = low_feats_list
        self.transform = transform
        self.high_patchs_list = []
        for low_idx, low_patch in enumerate(csv_file):
            high_patches = glob.glob(low_patch.replace('.jpeg', os.sep+'*.jpg')) + glob.glob(low_patch.replace('.jpeg', os.sep+'*.jpeg'))
            high_patches = high_patches + glob.glob(low_patch.replace('.jpg', os.sep+'*.jpg')) + glob.glob(low_patch.replace('.jpg', os.sep+'*.jpeg'))
            if len(high_patches) == 0:
                print('No valid patch extracted from: ' + self.low_patch_files[low_idx])
            for high_patch in high_patches:
                self.high_patchs_list.append((high_patch, low_idx))

    def __len__(self):
        return len(self.high_patchs_list)
    def __getitem__(self, idx):
        high_patch, low_idx = self.high_patchs_list[idx]
        high_patch = os.path.join(high_patch)
        high_patch = Image.open(high_patch)
        if self.transform:
            high_patch = self.transform(high_patch)
        return high_patch, self.low_feats_list[low_idx] 

class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        return {'input': img} 
    
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                    transform=Compose([
                                        ToTensor()
                                    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)

def compute_feats(args, bags_list, i_classifier, save_path=None, magnification='single'):
    i_classifier.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    device = "cuda:0" if args.use_cuda else "cpu"
    total_patch = 0
    for i in range(0, num_bags):
        feats_list = []
        if magnification=='single' or magnification=='low':            
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*.jpg')) + glob.glob(os.path.join(bags_list[i], '*.jpeg'))
        elif magnification=='high':
            csv_file_path = glob.glob(os.path.join(bags_list[i], '*'+os.sep+'*.jpg')) + glob.glob(os.path.join(bags_list[i], '*'+os.sep+'*.jpeg'))
            print("high patch")
        dataloader, bag_size = bag_dataset(args, csv_file_path)
        with torch.no_grad():
            t = tqdm(dataloader)
            t.set_description(f"Processing low patches: {i+1}/{num_bags}")
            for iteration, batch in enumerate(t):            
                patches = batch['input'].float().to(device)
                feats, classes = i_classifier(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)                
        if len(feats_list) == 0:
            print('No valid patch extracted from: ' + bags_list[i])
        else:
            total_patch += len(feats_list)
            df = pd.DataFrame(feats_list)
            os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
            df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1]+'.csv'), index=False, float_format='%.4f')
    print("total patch:", total_patch)

def compute_tree_feats(args, bags_list, embedder_low, embedder_high, save_path=None, fusion='fusion'):
    embedder_low.eval()
    embedder_high.eval()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor
    device = "cuda:0" if args.use_cuda else "cpu"  
    total_patch = 0  
    with torch.no_grad():
        for i in range(0, num_bags): 
            low_patches = glob.glob(os.path.join(bags_list[i], '*.jpg')) + glob.glob(os.path.join(bags_list[i], '*.jpeg'))
            feats_list = []
            feats_tree_list = []
            dataloader, bag_size = bag_dataset(args, low_patches)
            
            t = tqdm(dataloader)
            t.set_description(f"Processing low patches: {i+1}/{num_bags}")
            for iteration, batch in enumerate(t):            
                patches = batch['input'].float().to(device)                
                feats, classes = embedder_low(patches)
                feats = feats.cpu().numpy()
                feats_list.extend(feats)
                            
            high_dataset = HighBagDataset(low_patches, feats_list, transform=transforms.ToTensor())  
            high_dataloader = DataLoader(high_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
            high_patches_bar = tqdm(high_dataloader)
            high_patches_bar.set_description(f"Processing high patches {i+1}/{num_bags}")
            for iteration, batch in enumerate(high_patches_bar):            
                high_patchs, low_feats = batch                             
                high_patchs = high_patchs.float().to(device)
                feats, classes = embedder_high(high_patchs)                
                
                if fusion == 'fusion':                                      
                    feats = feats.cpu().numpy()+0.25*low_feats.numpy()                    
                elif fusion == 'cat':                    
                    feats = np.concatenate((feats.cpu().numpy(), 0.25*low_feats.numpy()), axis=-1)                    
                feats_tree_list.extend(feats)

            if len(feats_tree_list) == 0:
                print('No valid patch extracted from: ' + bags_list[i])
            else:
                total_patch += len(feats_tree_list)
                df = pd.DataFrame(feats_tree_list)
                os.makedirs(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2]), exist_ok=True)
                df.to_csv(os.path.join(save_path, bags_list[i].split(os.path.sep)[-2], bags_list[i].split(os.path.sep)[-1]+'.csv'), index=False, float_format='%.4f')
            print('\n')            
    print("total high patch:", total_patch)

def main():
    parser = argparse.ArgumentParser(description='Compute TCGA features from SimCLR embedder')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes [2]')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size of dataloader [128]')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of threads for datalodaer')
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0,), help='GPU ID(s) [0]')
    parser.add_argument('--backbone', default='resnet18', type=str, help='Embedder backbone [resnet18]')
    parser.add_argument('--norm_layer', default='instance', type=str, help='Normalization layer [instance]')
    parser.add_argument('--magnification', default='single', type=str, help='Magnification to compute features. Use `tree` for multiple magnifications. Use `high` if patches are cropped for multiple resolution and only process higher level, `low` for only processing lower level.')
    parser.add_argument('--weights', default=None, type=str, help='Folder of the pretrained weights, simclr/runs/*')
    parser.add_argument('--weights_high', default=None, type=str, help='Folder of the pretrained weights of high magnification, FOLDER < `simclr/runs/[FOLDER]`')
    parser.add_argument('--weights_low', default=None, type=str, help='Folder of the pretrained weights of low magnification, FOLDER <`simclr/runs/[FOLDER]`')
    parser.add_argument('--fusion', default='fusion', type=str, help='The fusion methods of two scale features')
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Dataset directory')
    parser.add_argument('--dataset', default='TCGA-lung-single', type=str, help='Dataset folder name [TCGA-lung-single]')
    parser.add_argument('--output_dir', default=None, type=str, help='')
    
    parser.add_argument('--use_cuda', action='store_true', help='whether uses the GPU')
    args = parser.parse_args()
    gpu_ids = tuple(args.gpu_index)
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)
    print(f"use CUDA: {args.use_cuda}")
    print(f"CUDA_VISIBLE_DEVICES:{os.environ['CUDA_VISIBLE_DEVICES']}")
    device = "cuda:0" if args.use_cuda else "cpu"   
    if not args.output_dir:
        args.output_dir = args.dataset

    if args.norm_layer == 'instance':
        norm=nn.InstanceNorm2d
        pretrain = False
    elif args.norm_layer == 'batch':  
        norm=nn.BatchNorm2d
        if args.weights == 'ImageNet':
            pretrain = True
        else:
            pretrain = False

    if args.backbone == 'resnet18':
        resnet = models.resnet18(pretrained=pretrain, norm_layer=norm)
        num_feats = 512
    if args.backbone == 'resnet34':
        resnet = models.resnet34(pretrained=pretrain, norm_layer=norm)
        num_feats = 512
    if args.backbone == 'resnet50':
        resnet = models.resnet50(pretrained=pretrain, norm_layer=norm)
        num_feats = 2048
    if args.backbone == 'resnet101':
        resnet = models.resnet101(pretrained=pretrain, norm_layer=norm)
        num_feats = 2048
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    
    if args.magnification == 'tree' and ((args.weights_high != None and args.weights_low != None) or args.weights != None):          
        i_classifier_h = mil.IClassifier(resnet, num_feats, output_class=args.num_classes).to(device)
        i_classifier_l = mil.IClassifier(copy.deepcopy(resnet), num_feats, output_class=args.num_classes).to(device)

        if args.weights_high == 'ImageNet' or args.weights_low == 'ImageNet' or args.weights== 'ImageNet':
            if args.norm_layer == 'batch':
                print('Use ImageNet features.')
            else:
                raise ValueError('Please use batch normalization for ImageNet feature')
        else:  
            # C16         
            if args.weights_high == 'mil_tcga_lung' or args.weights_low == 'mil_tcga_lung' or args.weights== 'mil_tcga_lung':
                weight_path = os.path.join('simclr', 'runs', 'mil_tcga_lung', 'model-v1.pth')                
            elif args.weights_high == 'mil_c16_v2'  or args.weights== 'mil_c16_v2':
                weight_path = os.path.join('simclr', 'runs', 'mil_c16', '20x', 'model-v2.pth') 
            elif args.weights_high == 'mil_c16_v1'  or args.weights== 'mil_c16_v1':
                weight_path = os.path.join('simclr', 'runs', 'mil_c16', '20x', 'model-v1.pth')
            elif args.weights_high == 'mil_c16_v0'  or args.weights== 'mil_c16_v0':
                weight_path = os.path.join('simclr', 'runs', 'mil_c16', '20x', 'model-v0.pth')  
            else:             
                weight_path = os.path.join('simclr', 'runs', args.weights_high, 'checkpoints', 'model.pth')
            
            print(f"Loading weights_high: {weight_path}")
            state_dict_weights = torch.load(weight_path)            
            for i in range(4):
                state_dict_weights.popitem()
            state_dict_init = i_classifier_h.state_dict()
            new_state_dict = OrderedDict()      
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier_h.load_state_dict(new_state_dict, strict=False)
            os.makedirs(os.path.join('embedder', args.output_dir), exist_ok=True)
            torch.save(new_state_dict, os.path.join('embedder', args.output_dir, 'embedder-high.pth'))
            
            if args.weights_high.startswith('mil_c16') and args.weights_low.startswith('mil_c16') :
                weight_path = os.path.join('simclr', 'runs', 'mil_c16', '5x', 'model.pth') 
            elif args.weights_high.startswith('mil_c16') and args.weights_low.startswith('c16_low'):
                model_name = "model_{}.pth".format(args.weights_low.split('_')[-1])
                weight_path = os.path.join('simclr', 'runs', 'Jun14_c16_low', 'checkpoints', model_name)
            else:
                weight_path = os.path.join('simclr', 'runs', args.weights_low, 'checkpoints', 'model.pth')
            print(f"Loading weights_low: {weight_path}")
            state_dict_weights = torch.load(weight_path)
            for i in range(4):
                state_dict_weights.popitem()
            state_dict_init = i_classifier_l.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier_l.load_state_dict(new_state_dict, strict=False)
            os.makedirs(os.path.join('embedder', args.output_dir), exist_ok=True)
            torch.save(new_state_dict, os.path.join('embedder', args.output_dir, 'embedder-low.pth'))
            print('Use pretrained features.')


    elif args.magnification == 'single' or args.magnification == 'high' or args.magnification == 'low':  
        i_classifier = mil.IClassifier(resnet, num_feats, output_class=args.num_classes).to(device)

        if args.weights == 'ImageNet':
            if args.norm_layer == 'batch':
                print('Use ImageNet features.')
            else:
                print('Please use batch normalization for ImageNet feature')
        else:            
            if args.magnification == 'high' and args.weights== 'mil_c16_v2':
                weight_path = os.path.join('simclr', 'runs', 'mil_c16', '20x', 'model-v2.pth') 
            elif args.magnification == 'high' and args.weights== 'mil_c16_v1':
                weight_path = os.path.join('simclr', 'runs', 'mil_c16', '20x', 'model-v1.pth')
            elif args.magnification == 'high' and args.weights== 'mil_c16_v0':
                weight_path = os.path.join('simclr', 'runs', 'mil_c16', '20x', 'model-v0.pth')
            elif args.magnification == 'low' and args.weights== 'mil_c16':
                weight_path = os.path.join('simclr', 'runs', 'mil_c16', '5x', 'model.pth')  
            elif args.weights is not None:
                weight_path = os.path.join('simclr', 'runs', args.weights, 'checkpoints', 'model.pth')
            else:
                weight_path = glob.glob('simclr/runs/*/checkpoints/*.pth')[-1]
            print("args.magnification:", args.magnification)
            print("args.weights:", args.weights)
            print(f"Loading single weight: {weight_path}")
            state_dict_weights = torch.load(weight_path)
            for i in range(4):
                state_dict_weights.popitem()
            state_dict_init = i_classifier.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            i_classifier.load_state_dict(new_state_dict, strict=False)
            os.makedirs(os.path.join('embedder', args.output_dir), exist_ok=True)
            torch.save(new_state_dict, os.path.join('embedder', args.output_dir, 'embedder.pth'))
            print('Use pretrained features.')
    
    if args.magnification == 'tree' or args.magnification == 'low' or args.magnification == 'high' :
        bags_path = os.path.join('..', args.dataset_dir, args.dataset, 'pyramid', '*', '*')
    else:
        bags_path = os.path.join('..', args.dataset_dir, args.dataset, 'single', '*', '*')
    feats_path = os.path.join('datasets', args.output_dir)
    os.makedirs(feats_path, exist_ok=True)
    bags_list = glob.glob(bags_path)  
    

    if args.magnification == 'tree':
        if args.fusion == 'fusion':
            print(f"Combine high and low feature by '{args.fusion}'")
            compute_tree_feats(args, bags_list, i_classifier_l, i_classifier_h, feats_path, 'fusion')
        elif args.fusion == 'cat':
            print(f"Combine high and low feature by '{args.fusion}'")
            compute_tree_feats(args, bags_list, i_classifier_l, i_classifier_h, feats_path, 'cat')
    else:           
        compute_feats(args, bags_list, i_classifier, feats_path, args.magnification)
    n_classes = glob.glob(os.path.join('datasets', args.output_dir, '*'+os.path.sep))
    n_classes = sorted(n_classes)
    print(os.path.join('datasets', args.output_dir, '*'+os.path.sep))
    print(f"n_classes:{n_classes}")
    all_df = []
    for i, item in enumerate(n_classes):
        bag_csvs = glob.glob(os.path.join(item, '*.csv'))
        bag_df = pd.DataFrame(bag_csvs)
        bag_df['label'] = i
        if args.dataset.startswith("C16_dataset"):
            bag_df.to_csv(os.path.join('datasets', args.output_dir, item.split(os.path.sep)[3]+'.csv'), index=False)
        else:
            bag_df.to_csv(os.path.join('datasets', args.output_dir, item.split(os.path.sep)[2]+'.csv'), index=False)
        all_df.append(bag_df)
    bags_path = pd.concat(all_df, axis=0, ignore_index=True)
    bags_path = shuffle(bags_path)
    bags_path.to_csv(os.path.join('datasets', args.output_dir, args.output_dir.split('/')[-1]+'.csv'), index=False)    
    
if __name__ == '__main__':
    main()