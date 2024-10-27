# -*- coding: utf-8 -*-
"""
@author: liu
"""
import os
import time
import json
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error


from torch_geometric.loader import DataLoader

import numpy as np

#set_name = 'S4169'
set_name = 'S1131'
#set_name = 'S645'
print(set_name)
numworkersgiven = 0

if set_name == 'S4169':
    from create_dataset_S4169 import AA_Dataset_con, AA_Dataset_MT
    from model_abla import SFM_Net as Net
    batchsizegiven = 64
    batchsizegiventest = 64
if set_name == 'S1131':
    from create_dataset_S1131 import AA_Dataset_con, AA_Dataset_MT
    from model_abla import SFM_Net as Net
    batchsizegiven = 64
    batchsizegiventest = 64
if set_name == 'S645':
    from create_dataset_S645 import AA_Dataset_con, AA_Dataset_MT
    from model_abla_S645 import SFM_Net as Net
    batchsizegiven = 256
    batchsizegiventest = 256
    
testmode = 'cross-validation'
#testmode = 'Ablastu01'
if  testmode == 'cross-validation':
    testmode = ''

elif testmode == 'Ablastu01':
    testmode = testmode
    if set_name == 'S4169':
        from model_abla import SFM_Netablastu01 as Net
    if set_name == 'S1131':
        from model_abla import SFM_Netablastu01 as Net

    if set_name == 'S645':
        from model_abla_S645 import SFM_Netablastu01 as Net


torch.backends.cudnn.benchmark = True


torch.manual_seed(0)

def root_mean_squared_error(true, pred):  
    squared_error = np.square(true - pred)   
    sum_squared_error = np.sum(squared_error)  
    rmse_loss = np.sqrt(sum_squared_error / true.size)  
    return rmse_loss  


def split_and_get_num(input_list):
    result_list = []
    for item in input_list:
        parts = item.split('__')
        result_list.append(int(parts[1])-1) 
    return result_list


def save_txt_file(results, save_path, output_filename):
    with open(output_filename, 'w') as txt:
        for ls in results:
            for line in ls:
                txt.write(f"{line}")

# json
data_dir_split = './data_split/'
data_split_mode = 'CV10_random'
# seed for retrieving corresponding downstream dataset splitting
splitting_seed = 256
# indicate the MT PPI complex source for retrieving corresponding data source file
mutation_source = '_foldx'

with open(data_dir_split + f'{set_name}_{data_split_mode}_data_split_{splitting_seed}.jsonl') as f:
    splits = f.readlines()
split_list = []
for split in splits:
    split_list.append(json.loads(split))

results = []
best_results = []
best_matrices = []
 
# dataset
dataset_WT = AA_Dataset_con(root = f'data/data_{set_name}/data_per_mutation_con/')
dataset_MT = AA_Dataset_MT(root = f'data/data_{set_name}/data_per_mutation_MT/')

#for fold in range(1, 2):
for fold in range(len(split_list)):
    print('current fold:', fold + 1)
    dataset_splits = split_list[fold]
    train_list, val_list = dataset_splits['train'], dataset_splits['val']
    train_num_ls, val_num_ls = split_and_get_num(train_list), split_and_get_num(val_list) 
    
    train_dataset_WT, test_dataset_WT = dataset_WT[train_num_ls], dataset_WT[val_num_ls]
    train_dataset_MT, test_dataset_MT = dataset_MT[train_num_ls], dataset_MT[val_num_ls] 
    
    batch_size_given = batchsizegiven
    batch_size_given_test = batchsizegiventest

    num_workers_given = numworkersgiven
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    # model def
    model = Net().to(device)
    model.load_state_dict(torch.load(f'./results/{set_name}_SFM_Net{testmode}/fold_{fold+1}_bestmodel.pth'))
    
    testset_wt = test_dataset_WT#[test_list_shuffled]
    testset_mt = test_dataset_MT#[test_list_shuffled]
    
    # DataLoader
    trainbatchsize, testbatchsize = batch_size_given, batch_size_given_test #32, 32
   
    test_loader_wt  = DataLoader(testset_wt,  batch_size = testbatchsize,  shuffle=False, num_workers= num_workers_given)
    test_loader_mt  = DataLoader(testset_mt,  batch_size = testbatchsize,  shuffle=False, num_workers= num_workers_given)

    print("-------Fold {}--- -------".format(fold + 1))

    
    # test
    model.eval()
    y_pred_test, y_real_test = [], []

    with torch.no_grad():
        for batch_test_wt, batch_test_mt in zip(test_loader_wt, test_loader_mt):
            batch_real_test = batch_test_wt.y.numpy()
            y_real_test.append(batch_test_wt.y)
            
            batch_test_wt = batch_test_wt.to(device)
            batch_test_mt = batch_test_mt.to(device)
            
            outputs = model(batch_test_wt, batch_test_mt)
            batch_test_y = batch_test_wt.y.to(device)

            # 
            batch_pred_test = outputs.cpu().detach().numpy()
            y_pred_test.append(batch_pred_test)

            MSE_batch_tt  = mean_squared_error(batch_real_test, batch_pred_test)
            RMSE_batch_tt = np.sqrt(MSE_batch_tt)
            MAE_batch_tt  = mean_absolute_error(batch_real_test, batch_pred_test)
            #RMSE_batch_tt = root_mean_squared_error(batch_real_test, batch_pred_test)
            pccs_batch_tt = pearsonr(batch_real_test, batch_pred_test)[0]
            #print(f"batch test--MSE: {MSE_batch_tt:.4f}, RMSE: {RMSE_batch_tt:.4f}, MAE: {MAE_batch_tt:.4f}, Pearson: {pccs_batch_tt:.4f}")

    # 
    all_real_tt = np.concatenate(y_real_test)
    all_pred_tt = np.concatenate(y_pred_test)
    
    MSE_tt  = mean_squared_error(all_real_tt, all_pred_tt)
    RMSE_tt = np.sqrt(MSE_tt)
    MAE_tt  = mean_absolute_error(all_real_tt, all_pred_tt)
    #RMSE_tt = root_mean_squared_error(all_real_tt, all_pred_tt)
    pccs_tt = pearsonr(all_real_tt.reshape(-1), all_pred_tt.reshape(-1))[0]
    epoch_result_test = f'MSE: {MSE_tt:.4f}, RMSE: {RMSE_tt:.4f}, MAE: {MAE_tt:.4f}, Pearson: {pccs_tt:.4f}'
    print("test:", epoch_result_test)  
    
    
    best_matrices.append([MSE_tt, RMSE_tt, MAE_tt, pccs_tt])


# 
best_matrices = np.array(best_matrices)
best_matrices_mean = np.mean(best_matrices, axis = 0) 
best_matrices_mean_result = f'--- Mean result of {len(best_matrices)} folds:---\ntest--MSE: {best_matrices_mean[0]:.4f}, RMSE: {best_matrices_mean[1]:.4f},\
 MAE: {best_matrices_mean[2]:.4f}, Pearson: {best_matrices_mean[3]:.4f}'
print(best_matrices_mean_result)
