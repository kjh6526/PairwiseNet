import os, json
import torch
import numpy as np
# import random
# from random import randrange
from torch.utils import data
import open3d as o3d
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

from envs import get_env

class Pcd(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.root = kwargs['root']
        self.batch_size = kwargs.get('batch_size', 100)
        self.random_pcds = kwargs.get('random_pcds', True)
        self.n_pcd = kwargs.get('n_pcd', 100)

        self.rand_pcds = []
        
        for dirpath, _, filenames in os.walk(self.root):
            for filename in filenames:
                if filename.endswith('.obj'):
                    mesh = o3d.io.read_triangle_mesh(os.path.join(dirpath, filename))
                    pcd = mesh.sample_points_uniformly(number_of_points=100*self.n_pcd)
                    pcd = torch.tensor(np.array(pcd.points), dtype=torch.float).T
                    self.rand_pcds.append(pcd)
    
    def __getitem__(self, idx):
        x = None
        j = torch.randint(len(self.rand_pcds), (1,))
        x = self._get_random_pcd(j)#.reshape(1, 3, self.n_pcd)
        return x

    def __len__(self):
        return self.batch_size   
    
    def _get_random_pcd(self, idx):
        # pcd_object = self.meshes[idx].sample_points_uniformly(number_of_points=self.n_pcd)
        # pcd_object = torch.tensor(np.array(pcd_object.points), dtype=torch.float).T
        pcd_object = self.rand_pcds[idx][:,torch.randperm(self.rand_pcds[idx].shape[1])[:self.n_pcd]]
        return pcd_object