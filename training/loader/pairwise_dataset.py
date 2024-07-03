import os, json
import torch
import numpy as np
from torch.utils import data
import pybullet as p
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

from envs import get_env

class Pairwise(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.root = kwargs['root']
        self.random_pcds = kwargs.get('random_pcds', True)
        self.n_pcd = kwargs.get('n_pcd', 100)
        
        split = kwargs['split']
        
        pair_indices = torch.load(os.path.join(self.root, 'pair_indices.pt')).type(torch.int)
        SE3 = torch.load(os.path.join(self.root, 'T_12.pt')).type(torch.float)
        y = torch.load(os.path.join(self.root, 'distances.pt')).type(torch.float)
        
        with open(os.path.join(self.root, 'Mid2mesh_dict.json'), 'r') as f:
            Mid2mesh_dict = json.load(f)
        Mid2mesh_dict = {int(k):v for k,v in Mid2mesh_dict.items()}
        
        SE3 = SE3[:, :3, :].view(-1, 12)
        
        n_objects = len(Mid2mesh_dict)
        self.pcds = [None] * n_objects
        self.meshes = [None] * n_objects
        self.rand_pcds = [None] * n_objects
        
        for Mid, meshname in Mid2mesh_dict.items():
            pcd_file = os.path.join(self.root, 'mesh_data', f'{meshname}_pcd_{self.n_pcd}.pt')
            self.pcds[Mid] = torch.load(pcd_file).type(torch.float)
            
            meshfile = os.path.join(self.root, 'mesh_data', meshname)
            self.meshes[Mid] = o3d.io.read_triangle_mesh(meshfile)
            
            pcd_object = self.meshes[Mid].sample_points_uniformly(number_of_points=self.n_pcd*100)
            pcd_object = torch.tensor(np.array(pcd_object.points), dtype=torch.float).T
            self.rand_pcds[Mid] = pcd_object
        
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
        if self.random_pcds:
            pcd1 = self._get_random_pcd(pair[0])
            pcd2 = self._get_random_pcd(pair[1])
        else:
            pcd1 = self.pcds[pair[0]]
            pcd2 = self.pcds[pair[1]]
        SE3 = self.SE3[idx]
        y = self.y[idx]
        return pcd1, pcd2, SE3, y
    
    def _get_random_pcd(self, idx):
        # pcd_object = self.meshes[idx].sample_points_uniformly(number_of_points=self.n_pcd)
        # pcd_object = torch.tensor(np.array(pcd_object.points), dtype=torch.float).T
        
        pcd_object = self.rand_pcds[idx][:, torch.randperm(self.rand_pcds[idx].shape[1])[:self.n_pcd]]
        return pcd_object