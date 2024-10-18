import torch
import numpy as np
import sys, os

import pybullet as p

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.subplots as sp
plotly_layout = dict(margin=dict(l=20, r=20, t=20, b=20))

from .samplers import UniformSampler, MCMCSampler, LangevinMCSampler

def get_augmenter(cfg, env, **kwargs):
    augmenter_cfg = cfg.get("augmenter")
    data_cfg = cfg.get("data").get('training')
    training_cfg = cfg.get("training")
    
    augmenter_name = augmenter_cfg.get('name', 'base')
    if augmenter_name == 'base':
        augmenter = BaseAugmenter(augmenter_cfg, data_cfg, env, **kwargs)
    elif augmenter_name == 'none':
        augmenter = NoneAugmenter()
    else:
        raise NotImplementedError
    return augmenter

def get_sampler(cfg):
    name = cfg['name']
    if name == 'uniform':
        sampler = UniformSampler(**cfg.get('params', {}))
    elif name == 'mcmc':
        sampler = MCMCSampler(**cfg.get('params', {}))
    elif name == 'lmc':
        sampler = LangevinMCSampler(**cfg.get('params', {}))
    else:
        raise NotImplementedError
    return sampler

class BaseAugmenter:
    def __init__(self, cfg, data_cfg, env, **kwargs):
        sampler_cfg = cfg.get('sampler')
        self.sampler = get_sampler(sampler_cfg)
        self.n_aug = cfg.get('n_augdata')
        self.replace = cfg.get('replace', False)
        self.env = env
        
    def augment(self, model, dls, **kwargs):
        
        train_dl = dls['training']
        valid_dl = dls['validation']
        
        n_aug_train = int(self.n_aug * 5/6)
        
        n_pairs = len(self.env.collision_pairs)
        n_joints = np.ceil(self.n_aug / n_pairs).astype(int)
        
        X = self.sampler.sample(n=n_joints, 
                                lb=self.env.q_min,
                                ub=self.env.q_max,
                                model=model,
                                **kwargs)
        
        pair_indices, SE3 = model.joint2pairwise(X)
        aug_indices = torch.randperm(n_joints*n_pairs)[:self.n_aug]
        pair_indices = pair_indices.view(n_joints*n_pairs, 2)[aug_indices].to(train_dl.dataset.y)
        SE3 = SE3.view(n_joints*n_pairs, 12)[aug_indices].to(train_dl.dataset.y)
        
        Y = self.env.calculate_distance_between_objects(X, self.env.collision_pairs, pbar=kwargs.get('pbar', False))
        Y = Y.view(n_joints*n_pairs, 1)[aug_indices].to(train_dl.dataset.y)
        
        mesh2Mid_dict = {}
        object_mesh_files = []
        for o_idx in range(self.env.n_objects):
            bID, lID = self.env.env_bullet.idx2id(o_idx)
            linkinfo = p.getVisualShapeData(bID)[lID+1]
            meshfile = linkinfo[4].decode('ascii')
            object_mesh_files.append(meshfile)
            if meshfile not in mesh2Mid_dict:
                mesh2Mid_dict[meshfile] = len(mesh2Mid_dict)

        mesh2Mid_map = np.vectorize(mesh2Mid_dict.get)

        # Oid  : Object ID, 0 ~ env.n_objects-1
        Oid2mesh_dict = dict(zip(np.arange(len(object_mesh_files)), object_mesh_files))    
        Oid2mesh_map = np.vectorize(Oid2mesh_dict.get)
        
        pair_meshes_np = Oid2mesh_map(pair_indices.cpu().numpy())
        pair_Mid_indices = torch.tensor( # pair_indices to be saved
            mesh2Mid_map(pair_meshes_np), 
            dtype=train_dl.dataset.pair_indices.dtype, 
            device=train_dl.dataset.pair_indices.device
        ) 
        
        if self.replace:
            N_train = len(train_dl.dataset.y)
            replace_idx = np.arange(N_train)
            np.random.shuffle(replace_idx)
            replace_idx = replace_idx[:n_aug_train]
            
            train_dl.dataset.pair_indices[replace_idx] = pair_Mid_indices[:n_aug_train]
            train_dl.dataset.SE3[replace_idx]          = SE3[:n_aug_train]
            train_dl.dataset.y[replace_idx]            = Y[:n_aug_train]
            
            N_valid = len(valid_dl.dataset.y)
            replace_idx = np.arange(N_valid)
            np.random.shuffle(replace_idx)
            replace_idx = replace_idx[:self.n_aug-n_aug_train]
            
            valid_dl.dataset.pair_indices[replace_idx] = pair_Mid_indices[n_aug_train:]
            valid_dl.dataset.SE3[replace_idx]          = SE3[n_aug_train:]
            valid_dl.dataset.y[replace_idx]            = Y[n_aug_train:]
        else:
            train_dl.dataset.pair_indices = torch.cat([train_dl.dataset.pair_indices, pair_Mid_indices[:n_aug_train]], dim=0)
            train_dl.dataset.SE3          = torch.cat([train_dl.dataset.SE3, SE3[:n_aug_train]], dim=0)
            train_dl.dataset.y            = torch.cat([train_dl.dataset.y, Y[:n_aug_train]], dim=0)
            
            valid_dl.dataset.pair_indices = torch.cat([valid_dl.dataset.pair_indices, pair_Mid_indices[n_aug_train:]], dim=0)
            valid_dl.dataset.SE3          = torch.cat([valid_dl.dataset.SE3, SE3[n_aug_train:]], dim=0)
            valid_dl.dataset.y            = torch.cat([valid_dl.dataset.y, Y[n_aug_train:]], dim=0)
        
        dls['training'] = train_dl
        dls['validation'] = valid_dl
        return dls #, pair_Mid_indices, SE3, Y
    
class NoneAugmenter:
    def __init__(self, **kwargs):
        pass
    def augment(self, dls, **kwargs):
        return dls