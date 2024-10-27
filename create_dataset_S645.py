import pandas as pd
#from rdkit import Chem
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np 
import os

print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

"""
!!!
NOTE: This file was replaced by dataset_featurizer.py
but is kept to illustrate how to build a custom dataset in PyG.
!!!
"""

# prot5 x[:,0:1024], dssp x[:,1024:1232], pssm x[:,1032:1052]
class AA_Dataset_con(Dataset):
    def __init__(self, root, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        #self.filename = filename
        super(AA_Dataset_con, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """

        return ['names_1_645.npy', 'targets_1_645.npy',\
                'edge_indexes_WT_1_645.npy', 'center_pls_WT_1_645.npy', \
                'features_prot5_WT_1_645.npy','features_dssp_WT_1_645.npy',\
                'features_pssm_WT_1_645.npy']
    
    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        #self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        #if self.test:
        #    return [f'data_test_{i}.pt' for i in list(self.data.index)]
        #else:
        #    return [f'data_{i}.pt' for i in list(self.data.index)]
        
        self.target_data = np.load(self.raw_paths[3],  allow_pickle=True) 
        #print(self.target_data, len(self.target_data))
        return [f'data_{i}.pt' for i in range(0, len(self.target_data))]

    def download(self):
        pass

    def process(self):
        name_data      = np.load(self.raw_paths[0],  allow_pickle=True)
        target_data    = np.load(self.raw_paths[1],  allow_pickle=True) 
        edge_data_WT   = np.load(self.raw_paths[2],  allow_pickle=True) 
        center_pls_WT  = np.load(self.raw_paths[3],  allow_pickle=True)
        # feature
        prot_data_WT   = np.load(self.raw_paths[4],  allow_pickle=True) 
        dssp_data_WT   = np.load(self.raw_paths[5],  allow_pickle=True)
        pssm_data_WT   = np.load(self.raw_paths[6],  allow_pickle=True)
        
        print(target_data)
        
        for idx in range(len(target_data)):
            name  = name_data[idx] 
            target  = torch.tensor(target_data[idx], dtype=torch.float) 
            #edge    = torch.tensor(edge_data[idx], dtype=torch.float)
            edge_data_1_WT = np.array(edge_data_WT[idx][0])
            edge_data_2_WT = np.array(edge_data_WT[idx][1])
            edge__WT = np.vstack((edge_data_1_WT,edge_data_2_WT))
            edge_WT = torch.tensor(edge__WT).to(torch.long)#.view(2, -1)
            # for center pooling
            center_pl_WT = torch.tensor(center_pls_WT[idx], dtype=torch.float)            
            
            #node_WT    = torch.tensor(node_data_WT[idx], dtype=torch.float)
            #node_MT    = torch.tensor(node_data_MT[idx], dtype=torch.float)
            x1 = torch.tensor(prot_data_WT[idx], dtype=torch.float)
            x2 = torch.tensor(dssp_data_WT[idx], dtype=torch.float)
            x3 = torch.tensor(pssm_data_WT[idx], dtype=torch.float)
            x_combined = torch.cat([x1, x2, x3], dim=1)


            # Create data object
            data = Data(x_combined, #x = [node_WT, node_MT], 
                        edge_index = edge_WT,
                        #edge_attr=edge_feats,
                        y=target,
                        center_pl = center_pl_WT,
                        name = name
                        ) 
            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, f'data_{idx}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data
#dataset = AA_Dataset_con(root = 'data_1217_S645/data_per_mutation_con/')


class AA_Dataset_MT(Dataset):
    def __init__(self, root, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        #self.filename = filename
        super(AA_Dataset_MT, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """

        return ['names_1_645.npy', 'targets_1_645.npy',\
                'edge_indexes_MT_1_645.npy', 'center_pls_MT_1_645.npy', \
                'features_prot5_MT_1_645.npy']
    
    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        #self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        #if self.test:
        #    return [f'data_test_{i}.pt' for i in list(self.data.index)]
        #else:
        #    return [f'data_{i}.pt' for i in list(self.data.index)]
        
        self.target_data = np.load(self.raw_paths[2],  allow_pickle=True) 
        #print(self.target_data, len(self.target_data))
        return [f'data_{i}.pt' for i in range(0, len(self.target_data))]

    def download(self):
        pass

    def process(self):
        name_data = np.load(self.raw_paths[0],  allow_pickle=True)
        target_data    = np.load(self.raw_paths[1],  allow_pickle=True) 
        edge_data_MT   = np.load(self.raw_paths[2],  allow_pickle=True)
        center_pls_MT  = np.load(self.raw_paths[3],  allow_pickle=True) 
        prot_data_MT   = np.load(self.raw_paths[4],  allow_pickle=True)

        print(target_data)
        
        
        for idx in range(len(target_data)):
            name  = name_data[idx]
            target  = torch.tensor(target_data[idx], dtype=torch.float) 
            node_MT    = torch.tensor(prot_data_MT[idx], dtype=torch.float)
            #edge    = torch.tensor(edge_data[idx], dtype=torch.float)
            edge_data_1_MT = np.array(edge_data_MT[idx][0])
            edge_data_2_MT = np.array(edge_data_MT[idx][1])
            edge__MT = np.vstack((edge_data_1_MT,edge_data_2_MT))
            edge_MT = torch.tensor(edge__MT).to(torch.long)#.view(2, -1)    
            # for center pooling
            center_pl_MT = torch.tensor(center_pls_MT[idx], dtype=torch.float)                       
            


            #print(center_pl_MT)
            # Create data object
            data = Data(x = node_MT , 
                        edge_index = edge_MT,
                        y=target,
                        center_pl = center_pl_MT,
                        name = name
                        ) 
            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, f'data_{idx}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        if self.test:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_test_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir, 
                                 f'data_{idx}.pt'))   
        return data
#dataset_MT = AA_Dataset_MT(root = 'data_1217_S645/data_per_mutation_MT/')


#print(dataset[0].x)
#print(dataset[0].y)
#print(dataset[0].edge_index)
#print(dataset[0].edge_index.t())
'''
for num in range(0,1282):
    i = dataset[num]
    if len(i.x) == len(i.y):
        pass
    else:
        print(num)
'''
'''
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
'''