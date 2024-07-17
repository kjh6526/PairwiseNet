import os
import sys
import time
import torch
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation as Rot
from training.model import get_model
import yaml
import argparse
from envs import get_env
from omegaconf import OmegaConf
from envs.lib.LieGroup import invSE3

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


if __name__ == '__main__':
    
    path = 'results/20240708-130759'
    cfg_path =  path + '/config_PairwiseNet_4arm_pretrain.yml'
    device = 0

    cfg = OmegaConf.load(cfg_path)
    env_cfg = OmegaConf.load(os.path.join(cfg.data.test.root, 'env_config.yml'))

   
    env = get_env(env_cfg, device=device, GUI = True)
    
    time_step = 0

    d_offset = 0.05
    d_nice = 0.30
    mu = np.log(2)/(d_nice-d_offset)

    model = get_model(cfg.model, env=env).to(device)
    checkpoint = torch.load(path+'/model_best_accuracy.pkl')
    model_state_dict = checkpoint['model_state']
    model.load_state_dict(model_state_dict)   
    pairwise = Pairwise2Global(model, cfg, env) 
    
    while True:
        current_joint, current_vel, _ = env.env_bullet.getJointStates()
        current_joint = torch.tensor(current_joint, dtype = torch.float, device = device, requires_grad = True)
        current_vel = torch.tensor(current_vel, dtype = torch.float, device = device)
        input_vec = current_joint.unsqueeze(0) 
        d_min = pairwise(input_vec).squeeze()
        gamma = torch.exp(mu*(d_min-d_offset))
        gamma.backward()
        gradient_gamma = current_joint.grad
        repulsion = gradient_gamma * torch.log(gamma-1) / (torch.norm(gradient_gamma)**2+0.01)
        if repulsion.isnan().any():
            raise ValueError("NaN value detected in repulsion. Stopping the simulation.")
        # print("d_min",d_min)
        # print("gamma", gamma)
        # print("grad", gradient_gamma)
        # print("qdot", current_vel)
        # print("q", current_joint)
        # print("repulsion", repulsion)
        # print("log(gamma-1)", torch.log(gamma-1))
        # print("dgamma/dt", torch.dot(gradient_gamma,repulsion))
        if gamma.item() > 2:
            target_vel = current_vel 
        else:
            target_vel = - 100*repulsion
        env.env_bullet.set2TargetVelocities(target_vel.tolist())
        p.stepSimulation()
        time_step += 1
        # time.sleep(1./240.) 