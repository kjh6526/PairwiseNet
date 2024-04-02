""" 
    Minimum distance learning between two mesh (pointcloud)

    input : Mesh 1 == (B, dim_pcd, num_pcd), Mesh 2 == (B, dim_pcd, num_pcd), flattened SE3 T == (B, 12) 
    output : minimum distance d == (B, 1)

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AUROC
get_auroc = AUROC(task='binary')

import os
from tqdm import tqdm, trange

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.subplots as sp
plotly_layout = dict(margin=dict(l=20, r=20, t=20, b=20))

from training.model.activations import get_activation
from envs import get_env
from envs.lib.LieGroup import invSE3

class PairwiseNet(nn.Module):
    def __init__(self, 
                encoder, 
                hidden_nodes=[128, 128, 128],
                activation='relu',
                last_activation='relu',
                output_dims=1, 
                **kwargs):
        super(PairwiseNet, self).__init__()
        
        self.encoder = encoder
        self.output_dims = output_dims
        self.start_dims = encoder.output_dims
        
        self.nodes = [self.start_dims*2 + 12] + list(hidden_nodes) + [self.output_dims]
        self.activation = get_activation(name=activation)
        
        self.last_activation = get_activation(name=last_activation)
        self.layers = torch.nn.ModuleList()
        
        for layer_idx in range(len(self.nodes)-1):
            self.layers.append(torch.nn.Linear(self.nodes[layer_idx], self.nodes[layer_idx+1]))
            
    def __call__(self, pcd1, pcd2, SE3):
        return self.forward(pcd1, pcd2, SE3)
    
    def forward(self, pcd1, pcd2, SE3):
        pcd1_embed = self.encoder(pcd1)
        pcd2_embed = self.encoder(pcd2)
        
        x = torch.cat([pcd1_embed, pcd2_embed, SE3], dim=1)
        
        for layer_idx in range(len(self.layers)-1):
            x = self.layers[layer_idx](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = self.last_activation(x)
        
        return x
    
    def forward_from_embed(self, pcd1_embed, pcd2_embed, SE3):
        
        x = torch.cat([pcd1_embed, pcd2_embed, SE3], dim=1)
        
        for layer_idx in range(len(self.layers)-1):
            x = self.layers[layer_idx](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = self.last_activation(x)
        
        return x
    
    def get_device(self):
        return list(self.parameters())[0].device

    def save(self, path):
        torch.save({'state_dict': self.state_dict()}, path)
        
    def train_step(self, pcd1, pcd2, SE3, y, criterion, optimizer, **kwargs):
        optimizer.zero_grad()
        output = self(pcd1, pcd2, SE3)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}
    
    def validation_step(self, pcd1, pcd2, SE3, y, criterion, **kwargs):
        output = self(pcd1, pcd2, SE3)
        loss = criterion(output, y)
        return {"loss": loss.item()}
    
    def eval_step(self, test_dl, env, cfg, **kwargs):
        device = kwargs['device']
        collision_thr = kwargs.get('collision_thr', 0.0)
        
        checker = Pairwise2Global(self, cfg, env)
        
        output = [None]*len(test_dl)
        target = [None]*len(test_dl)
        label  = [None]*len(test_dl)
        pred   = [None]*len(test_dl)

        b_idx = 0
        for x, y in tqdm(test_dl, disable=not kwargs.get('pbar', True), ncols=100, desc='eval'):
            o = checker(x.to(device)).squeeze().detach().cpu()
            p = (o < collision_thr).type(torch.int)
            output[b_idx] = o
            pred[b_idx] = p 

            lb = (y.squeeze() < collision_thr).type(torch.int)
            target[b_idx] = y.squeeze()
            label[b_idx] = lb.squeeze()
            b_idx += 1

        output = torch.cat(output)
        target = torch.cat(target)
        label = torch.cat(label)
        pred = torch.cat(pred)
        
        accu = ((output < collision_thr) == label).sum() / len(output)
        auroc = get_auroc(-output, label)
        
        safe_thr = output[label == 1].max()

        mse = torch.nn.MSELoss()(output, target)

        FP = ((output <= safe_thr) & (label == 0)).type(torch.int).sum()
        TN = ((output > safe_thr) & (label == 0)).type(torch.int).sum()
        safe_FPR = FP / (FP+TN)

        return {
            'eval/accuracy_': accu,
            'eval/AUROC_': auroc,
            'eval/mse_': mse,
            'eval/safe_FPR_': safe_FPR,
        }
        
    def visualization_step(self, **kwargs):
        return {}
    
class Pairwise2Global:
    def __init__(self, model, cfg, env, **kwargs):
        self.model = model
        self.cfg = cfg
        self.device = self.model.get_device()
        self.env = env
        
        self.collision_pairs = torch.tensor(env.collision_pairs)
        
        pcd1s = []
        pcd2s = []
        for b_idx in range(len(self.collision_pairs)):
            obj1_idx = self.collision_pairs[b_idx, 0]
            obj2_idx = self.collision_pairs[b_idx, 1]
            
            pcd1 = torch.load(os.path.join(self.cfg.data.test.root, 'pcds', f'pcd_{obj1_idx}.pt')).type(torch.float)
            pcd2 = torch.load(os.path.join(self.cfg.data.test.root, 'pcds', f'pcd_{obj2_idx}.pt')).type(torch.float)
            pcd1s.append(pcd1)
            pcd2s.append(pcd2)
            
        self.pcd1 = torch.stack(pcd1s, dim=0)
        self.pcd2 = torch.stack(pcd2s, dim=0)

        self.pcd1_embed = self.model.encoder(self.pcd1.to(self.device)).detach()
        self.pcd2_embed = self.model.encoder(self.pcd2.to(self.device)).detach()
        
        self.model.eval()
        
    def calculate_min_distance(self, X):
        assert self.env.n_dof == X.shape[1]
        
        n_data = len(X)
        SE3 = self.env.get_Ts_objects(X).to(self.device)

        n_pairs = len(self.collision_pairs)

        T_1 = SE3[:, self.collision_pairs[:, 0]].view(n_data*n_pairs, 4, 4)
        T_2 = SE3[:, self.collision_pairs[:, 1]].view(n_data*n_pairs, 4, 4)
        T_12 = invSE3(T_1) @ T_2

        pcd1_embed_repeated = self.pcd1_embed.unsqueeze(0).repeat_interleave(n_data, dim=0).view(n_data*n_pairs, -1)
        pcd2_embed_repeated = self.pcd2_embed.unsqueeze(0).repeat_interleave(n_data, dim=0).view(n_data*n_pairs, -1)
        SE3 = T_12[:, :3, :].view(-1, 12)
        
        prediction = self.model.forward_from_embed(pcd1_embed_repeated.to(self.device), pcd2_embed_repeated.to(self.device), SE3.to(self.device))
        prediction = prediction.view(n_data, n_pairs, 1)
        
        output = prediction.min(dim=1).values
        return output
    
    def __call__(self, X):
        return self.calculate_min_distance(X)