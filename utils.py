import torch
import os

def load_model(milnet, dataset, model, loss=None, model_weight=None):
    model_dir = 'weights'
    model_name = None
    if model == 'dsmil':    
        if dataset=='C16_dataset_c16_low99_v0': 
            weight_name = 'C16_dataset_c16_low99_v0_dsmil_1class'  
            weight_time = '20220929_183051'     # 0.86, 0.9332
            model_name = 'best.pth'
    
    elif model == 'frmil':
        if dataset=='C16_dataset_c16_low99_v0':                                          
            if loss=='FrmilLoss' or loss=='FrmilLoss2':                 
                weight_name = 'C16_dataset_c16_low99_v0_frmil_FrmilLoss_1class_thres10.17_dropout_0.2'
                weight_time = '20221103_165624'   # ACC05 0.8915 | AUC 0.9457  
    
    if model_weight is not None:
        model_path = os.path.join(model_dir, model_weight, 'best.pth')
    elif model_name is None:
        model_path = os.path.join(model_dir, weight_name, weight_time, 'best.pth')
    else:
        model_path = os.path.join(model_dir, weight_name, weight_time, model_name) 
    print("Loading the pretrain model from:", model_path)
    state_dict_weights = torch.load(model_path)    
    milnet.load_state_dict(state_dict_weights)
    return milnet

def print_result(avg_score, avg_05_score, aucs, praucs, result_type='', dataset=None): 
    if dataset=='Lung': 
        print(f"The {result_type} Model: ACC {avg_score:.4f} | ACC05 {avg_05_score:.4f} |auc_LUAD: {aucs[0]:.4f}/{praucs[0]:.4f}, auc_LUSC: {aucs[1]:.4f}/{praucs[1]:.4f}")        
    else: 
        print(f"The {result_type} Model: ACC {avg_score:.4f} | ACC05 {avg_05_score:.4f} | AUC {aucs[0]:.4f}/{praucs[0]:.4f}")
        
