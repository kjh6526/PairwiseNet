import os
import torch
from torch.utils import data

from envs import get_env

class Pairwise(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.root = kwargs['root']
        
        split = kwargs['split']
        
        pair_indices = torch.load(os.path.join(self.root, 'pair_indices.pt')).type(torch.int)
        SE3 = torch.load(os.path.join(self.root, 'T_12.pt')).type(torch.float)
        y = torch.load(os.path.join(self.root, 'distances.pt')).type(torch.float)
        
        SE3 = SE3[:, :3, :].view(-1, 12)
        
        n_objects = len(os.listdir(os.path.join(self.root, 'pcds')))
        self.pcds = [None] * n_objects
        for pcd_file in os.listdir(os.path.join(self.root, 'pcds')):
            object_idx = int(pcd_file.split('_')[1].split('.')[0])
            self.pcds[object_idx] = torch.load(os.path.join(self.root, 'pcds', pcd_file)).type(torch.float)
        
        split_train_val_test = (5/7, 1/7, 1/7)
        num_train_data = int(len(y) * split_train_val_test[0])
        num_valid_data = int(len(y) * split_train_val_test[1]) 
        
        idx = torch.arange(len(y))
        if split == "training":
            idx = idx[:num_train_data]
        elif split == "validation":
            idx = idx[num_train_data:num_train_data + num_valid_data]
        elif split == "test":
            idx = idx[num_train_data + num_valid_data:]
        elif split == "all":
            pass

        self.pair_indices = pair_indices[idx]
        self.SE3 = SE3[idx]
        self.y = y[idx]

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        pair = self.pair_indices[idx]
        pcd1 = self.pcds[pair[0]]
        pcd2 = self.pcds[pair[1]]
        SE3 = self.SE3[idx]
        y = self.y[idx]
        return pcd1, pcd2, SE3, y