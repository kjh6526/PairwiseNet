import copy
import os
import sys
from dis import dis
from math import degrees

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.subplots as sp
import torch
from tqdm import tqdm, trange

plotly_layout = dict(margin=dict(l=20, r=20, t=20, b=20))

import fcl, hppfcl
from scipy.spatial.transform import Rotation as Rot

from envs.models.panda.multipanda_bullet import MultiPanda_bullet
from envs.models.panda.panda import Panda

class MultiPanda:
    def __init__(self, base_poses, base_orientations, **kwargs):
        
        device = kwargs.get('device', 'cpu')
        
        self.name = 'multipanda'
        
        self.robots = []
        self.n_robot = len(base_poses)
        self.n_dof = 0
        
        self.q_min = []
        self.q_max = []
        self.qdot_min = []
        self.qdot_max = []
        
        self.T_bases = []
        
        self.collision_shape = kwargs.get('collision_shape', 'mesh')
        self.mesh_type = kwargs.get('mesh_type', 'simplified')
        
        for r_idx in range(self.n_robot):
            ri_base_pos = base_poses[r_idx]
            ri_base_ori = base_orientations[r_idx]
            ri_T_base = torch.eye(4)
            ri_T_base[:3, :3] = torch.as_tensor(Rot.from_euler('XYZ', [0, 0, ri_base_ori]).as_matrix())
            ri_T_base[:3, 3] = torch.as_tensor(ri_base_pos)
            self.T_bases.append(ri_T_base.to(device))
            self.robots.append(Panda(T_base=ri_T_base, device=device, collision_shape=self.collision_shape, mesh_type=self.mesh_type))
            
            
            self.n_dof += self.robots[-1].n_dof
            self.q_min.append(self.robots[-1].q_min.squeeze())
            self.q_max.append(self.robots[-1].q_max.squeeze())
            self.qdot_min.append(self.robots[-1].qdot_min.squeeze())
            self.qdot_max.append(self.robots[-1].qdot_max.squeeze())
            
        self.q_min = torch.cat(self.q_min)
        self.q_max = torch.cat(self.q_max)
        self.qdot_min = torch.cat(self.qdot_min)
        self.qdot_max = torch.cat(self.qdot_max)
        
        self.q = torch.zeros(self.n_dof)
        self.device = device
        self.mesh_divide = True
        
        self.env_bullet = MultiPanda_bullet(base_poses, base_orientations, **kwargs)
        
        self.plot_width = 800
        self.plot_height = 800
        
    def set_inputs(self, q):
        assert len(q) == self.n_dof
        self.q = torch.as_tensor(q).to(self.device)
        self.env_bullet.reset2TargetPositions(self.q.detach().cpu())
        
    def get_Ts(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
            
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        N = x.shape[0]
        
        assert x.shape[1] == self.n_dof, f'DOF of the system and input mismatch: DOF=={self.n_dof}, got {x.shape[1]}'
        
        x = x.to(self.device)
        T_sj = []
        start_idx = 0
        for r_idx in range(self.n_robot):
            _, ri_Tsj = self.robots[r_idx].solveBatchForwardKinematics(x[:, start_idx:start_idx+self.robots[r_idx].n_dof], return_T_link=True)
            start_idx += self.robots[r_idx].n_dof
            T_sj.append(ri_Tsj)
        
        output = torch.cat(T_sj, dim=1)
                
        return output
    
    def get_image(self, **kwargs):
        img = self.env_bullet.get_image(self.q, **kwargs)
        return img
    
    def get_mindist(self, mode='bullet'):
        
        if mode == 'fcl':
            fcl_mgrs = []
            start_idx = 0
            for r_idx in range(self.n_robot):
                tmp_mgr = fcl.DynamicAABBTreeCollisionManager()
                tmp_mgr.registerObjects(self.robots[r_idx].fcl_objs(jointPos=self.q[start_idx:start_idx+self.robots[r_idx].n_dof]))
                fcl_mgrs.append(tmp_mgr)
                start_idx += self.robots[r_idx].n_dof
                
            min_distance = 1e10
            for i_idx in range(self.n_robot):
                for j_idx in range(i_idx+1, self.n_robot):
                    tmp_mgr1 = fcl_mgrs[i_idx]
                    tmp_mgr2 = fcl_mgrs[j_idx]
                    
                    tmp_mgr1.setup()
                    tmp_mgr2.setup()
                    
                    req = fcl.CollisionRequest(enable_contact=True, enable_cost=True)
                    rdata = fcl.CollisionData(request = req)
                    tmp_mgr1.collide(tmp_mgr2, rdata, fcl.defaultCollisionCallback)
                    
                    isCollision = rdata.result.is_collision
                    
                    if isCollision:
                        max_pd_depth = -1e10
                        for contact in rdata.result.contacts:
                            if contact.penetration_depth > max_pd_depth:
                                max_pd_depth = contact.penetration_depth
                                
                        distance = -max_pd_depth
                        
                    else:
                        req = fcl.DistanceRequest(enable_nearest_points=True)
                        ddata = fcl.DistanceData(request = req)
                        tmp_mgr1.distance(tmp_mgr2, ddata, fcl.defaultDistanceCallback)
                        
                        distance = ddata.result.min_distance
                        
                    if min_distance > distance:
                        min_distance = distance
                        
        if mode == 'hppfcl':
            hppfcl_mgrs = []
            start_idx = 0
            for r_idx in range(self.n_robot):
                tmp_mgr = hppfcl.DynamicAABBTreeCollisionManager()
                tmp_objs = self.robots[r_idx].hppfcl_objs(jointPos=self.q[start_idx:start_idx+self.robots[r_idx].n_dof])
                for obj in tmp_objs:
                    tmp_mgr.registerObject(obj)
                hppfcl_mgrs.append(tmp_mgr)
                start_idx += self.robots[r_idx].n_dof

            min_distance = 1e10
            for i_idx in range(self.n_robot):
                for j_idx in range(i_idx+1, self.n_robot):
                    tmp_mgr1 = hppfcl_mgrs[i_idx]
                    tmp_mgr2 = hppfcl_mgrs[j_idx]

                    tmp_mgr1.setup()
                    tmp_mgr2.setup()
                    rdata = hppfcl.CollisionCallBackDefault()
                    tmp_mgr1.collide(tmp_mgr2, rdata)

                    isCollision = rdata.data.result.isCollision()

                    if isCollision:
                        max_pd_depth = -1e10
                        for contact in rdata.data.result.getContacts():
                            if contact.penetration_depth > max_pd_depth:
                                max_pd_depth = contact.penetration_depth

                        distance = -max_pd_depth

                    else:
                        ddata = hppfcl.DistanceCallBackDefault()
                        tmp_mgr1.distance(tmp_mgr2, ddata)

                        distance = ddata.data.result.min_distance

                    if min_distance > distance:
                        min_distance = distance
                        
        elif mode == 'bullet':
            min_distance = self.env_bullet.check_collision(self.q.detach().cpu())
            
        return min_distance
        
    def calculate_min_distance(self, X, pbar=False, mode='bullet'):
        # X : (batch, N_DOF)
        if isinstance(X, torch.Tensor):
            output = torch.zeros(len(X), 1).to(X).type(torch.float)
            X = X.detach().cpu().numpy()
        elif isinstance(X, np.ndarray):
            output = np.zeros((len(X), 1), dtype=np.float32)
        else:
            X = torch.tensor(X).type(torch.float)
            output = torch.zeros(len(X), 1).to(X).type(torch.float)
            
        for b_idx in trange(len(X), disable=not pbar, desc='Min.dist', ncols=100):
            self.set_inputs(X[b_idx])
            output[b_idx] = self.get_mindist(mode=mode)
            
        return output
    
    def calculate_distance_between_links(self, X, link_pairs, pbar=False):
        if isinstance(X, torch.Tensor):
            output = torch.zeros(len(X), len(link_pairs), 1).to(X).type(torch.float)
            X = X.detach().cpu().numpy()
        elif isinstance(X, np.ndarray):
            output = np.zeros((len(X), len(link_pairs), 1))
            
        for b_idx in trange(len(X), disable=not pbar, desc='Min.dist', ncols=100):
            self.set_inputs(X[b_idx])
            for ls_idx, links in enumerate(link_pairs):
                output[b_idx, ls_idx] = self.env_bullet.get_distance_between_links(links[0], links[1])
                
        return output
    
    def plot(self):
        img = self.get_image(width=1280, height=720, yaw=45)
        fig = go.Figure(go.Image(z=img)).update_layout(**plotly_layout, width=1280, height=720)
        return fig
    
    