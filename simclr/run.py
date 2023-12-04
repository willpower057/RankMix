from simclr import SimCLR
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import os, glob
import pandas as pd
import argparse

def generate_csv(args):
    if args.level=='high' and args.multiscale==1:
        path_temp = os.path.join('..', '..', args.dataset_dir, args.dataset, 'pyramid', '*', '*', '*', '*.jpeg')
        patch_path = glob.glob(path_temp) # /class_name/bag_name/5x_name/*.jpeg
    if args.level=='low' and args.multiscale==1:
        path_temp = os.path.join('..', '..', args.dataset_dir, args.dataset, 'pyramid', '*', '*', '*.jpeg')
        patch_path = glob.glob(path_temp) # /class_name/bag_name/*.jpeg
    if args.multiscale==0:
        path_temp = os.path.join('..', '..', args.dataset_dir, args.dataset, 'single', '*', '*', '*.jpeg')
        patch_path = glob.glob(path_temp) # /class_name/bag_name/*.jpeg
    df = pd.DataFrame(patch_path)
    df.to_csv('all_patches.csv', index=False)
    print("all_patches:", len(df))
    # raise
        
def generate_usibility_csv(args):
    slide_paths = pd.read_csv('HistoQC_reader_train.csv').iloc[:,0]    
    print(f"Loading slide_paths: {'HistoQC_reader_train.csv'}")      
    print(len(slide_paths), slide_paths)
    slide_names = [slide_path.split('/')[-1].rstrip('.csv') for slide_path in slide_paths]
    print(slide_names[:5])
    
    all_train_df = []
    for slide_name in slide_names:
        if args.level=='high' and args.multiscale==1:
            path_temp = os.path.join('..', '..', args.dataset_dir, args.dataset, 'pyramid', '*', slide_name, '*', '*.jpeg')
            patch_path = glob.glob(path_temp) # /class_name/bag_name/5x_name/*.jpeg
        if args.level=='low' and args.multiscale==1:
            path_temp = os.path.join('..', '..', args.dataset_dir, args.dataset, 'pyramid', '*', slide_name, '*.jpeg')
            patch_path = glob.glob(path_temp) # /class_name/bag_name/*.jpeg
        if args.multiscale==0:
            raise NotImplementedError("not implement multiscale==0")
            patch_path = glob.glob(path_temp) # /class_name/bag_name/*.jpeg
        all_train_df.append(pd.DataFrame(patch_path))
    train_bags_path = pd.concat(all_train_df, axis=0, ignore_index=True)
    train_bags_path.to_csv('all_patches.csv', index=False)
    print("train_bags_path:", len(train_bags_path))
    # raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, default='low', help='Magnification level to compute embedder (low/high)')
    parser.add_argument('--multiscale', type=int, default=0, help='Whether the patches are cropped from multiscale (0/1-no/yes)')
    parser.add_argument('--dataset_dir', type=str, default='dataset', help='Dataset directory')
    parser.add_argument('--dataset', type=str, default='HistoQC_reader', help='Dataset folder name')
    args = parser.parse_args()
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    gpu_ids = eval(config['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpu_ids)   
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])         
    print("Generate all_patches.csv ...")
    if args.dataset == 'HistoQC_reader':
        generate_usibility_csv(args)    
    else:
        generate_csv(args)
    simclr = SimCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":
    main()
