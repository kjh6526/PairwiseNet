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

import pybullet as p
import open3d as o3d
import fcl, hppfcl
from scipy.spatial.transform import Rotation as Rot

from envs.models.panda.multipanda_bullet import MultiPanda_bullet
from envs.models.panda.panda import Panda

class MultiPanda:
    def __init__(self, base_poses, base_orientations, obstacles=None, **kwargs):
        
        self.env_bullet = MultiPanda_bullet(base_poses, base_orientations, obstacles, **kwargs)
        self.n_objects = self.env_bullet.n_objects
        self.collision_pairs = self.env_bullet.collision_pairs
        
        self.name = 'multipanda'
        device = kwargs.get('device', 'cpu')
        self.device = device
        
        # Robots
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
        self.hand = kwargs.get('hand', False)
        
        for r_idx in range(self.n_robot):
            ri_base_pos = base_poses[r_idx]
            ri_base_ori = base_orientations[r_idx]
            ri_T_base = torch.eye(4)
            ri_T_base[:3, :3] = torch.as_tensor(Rot.from_euler('XYZ', [0, 0, ri_base_ori]).as_matrix())
            ri_T_base[:3, 3] = torch.as_tensor(ri_base_pos)
            self.T_bases.append(ri_T_base.to(device))
            self.robots.append(Panda(T_base=ri_T_base, hand=self.hand, device=device, collision_shape=self.collision_shape, mesh_type=self.mesh_type))
            
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
        
        # Obstacles
        self.obstacles = []
        for body_info in self.env_bullet.bodies:
            if body_info['type'] == 'obstacle':
                tmpfclCollisionObjects = []
                tmphppfclCollisionObjects = []
        
                bID = body_info['id']
                
                T_links = []
                vertices = []
                triangles = []
                for lID in body_info['links']:
                    object_info = p.getVisualShapeData(bID)[lID+1]
                    mesh = o3d.io.read_triangle_mesh(object_info[4])
                    tmpV = np.asarray(mesh.vertices)
                    tmpT = np.asarray(mesh.triangles)
                    
                    vertices.append(tmpV)
                    triangles.append(tmpT)
                    
                    if lID == -1:
                        object_state = p.getBasePositionAndOrientation(bID)
                    else:
                        object_state = p.getLinkState(bID, lID, 0, 0)
                        
                    T_link = torch.eye(4)
                    T_link[:3, :3] = torch.tensor(Rot.from_quat(object_state[1]).as_matrix())
                    T_link[:3, 3] = torch.tensor(object_state[0])
                    T_links.append(T_link)
                    
                    # FCL setting
                    tmpshape = fcl.BVHModel()
                    tmpshape.beginModel(len(tmpV), len(tmpT))
                    tmpshape.addSubModel(tmpV, tmpT)
                    tmpshape.endModel()
                    tmpobj = fcl.CollisionObject(tmpshape, fcl.Transform())

                    tmp_transform = fcl.Transform(Rot.from_quat(object_state[1]).as_matrix(), np.array(object_state[0]))
                    tmpobj.setTransform(tmp_transform)
                    tmpfclCollisionObjects.append(tmpobj)
                    
                    # HPP-FCL setting
                    V_hpp = hppfcl.StdVec_Vec3f()
                    T_hpp = hppfcl.StdVec_Triangle()
                    V_hpp.extend([tmpV[i] for i in range(len(tmpV))])
                    for idx in range(len(tmpT)):
                        T_hpp.append(hppfcl.Triangle(tmpT[idx, 0].item(), tmpT[idx, 1].item(), tmpT[idx, 2].item()))
                    
                    hppobj = hppfcl.CollisionObject(hppfcl.Convex(V_hpp, T_hpp))
                    
                    tmp_transform = hppfcl.Transform3f.Identity()
                    tmp_transform.setRotation(Rot.from_quat(object_state[1]).as_matrix())
                    tmp_transform.setTranslation(np.array(object_state[0]))
                    
                    hppobj.setTransform(tmp_transform)
                    tmphppfclCollisionObjects.append(hppobj)
                    
                body_info['T_links'] = torch.stack(T_links, dim=0)
                body_info['vertices'] = vertices
                body_info['triangles'] = triangles
                body_info['fcl_objs'] = tmpfclCollisionObjects
                body_info['hppfcl_objs'] = tmphppfclCollisionObjects
                self.obstacles.append(body_info)
        
        
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
    
    def get_Ts_objects(self, x, object_indices='all'):
        
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float)
            
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        N = x.shape[0]
        
        x = x.to(self.device)
        T_sj = []
        start_idx = 0
        for r_idx in range(self.n_robot):
            _, ri_Tsj = self.robots[r_idx].solveBatchForwardKinematics(x[:, start_idx:start_idx+self.robots[r_idx].n_dof], return_T_link=True)
            start_idx += self.robots[r_idx].n_dof
            T_sj.append(ri_Tsj)
        
        T_objects = []
        object_indices = list(range(self.n_objects))
        
        for obj_idx in object_indices:
            assert obj_idx in list(range(self.n_objects)), f"Invalid object index: got {obj_idx}, possible 0 ~ {self.n_objects-1}"
            bodyID, linkID = self.env_bullet.idx2id(obj_idx)
            body_info = self.env_bullet.body_dict[bodyID]
            
            if body_info['type'] == 'robot':
                robot_idx = int(body_info['name'].split('_')[-1])
                if linkID == -1:
                    T_objects.append(self.T_bases[robot_idx].unsqueeze(0).unsqueeze(1).repeat_interleave(N, dim=0))
                else:
                    T_objects.append(T_sj[robot_idx][:, [linkID], :, :].to(self.device))
                    
            else:
                obs_index = int(body_info['name'].split('_')[-1])    
                T_object = body_info['T_links'][linkID+1].unsqueeze(0).unsqueeze(1).repeat_interleave(N, dim=0)
                T_objects.append(T_object.to(self.device))
        
        T_objects = torch.cat(T_objects, dim=1)
        
        return T_objects
    
    def get_image(self, **kwargs):
        img = self.env_bullet.get_image(self.q, **kwargs)
        return img
    
    def get_mindist(self, mode='bullet'):
        
        if mode == 'fcl':
            fcl_mgrs = []
            mgrs_type = []
            start_idx = 0
            for r_idx in range(self.n_robot):
                tmp_mgr = fcl.DynamicAABBTreeCollisionManager()
                tmp_mgr.registerObjects(self.robots[r_idx].fcl_objs(jointPos=self.q[start_idx:start_idx+self.robots[r_idx].n_dof]))
                fcl_mgrs.append(tmp_mgr)
                mgrs_type.append('robot')
                start_idx += self.robots[r_idx].n_dof
                
            for obstacle in self.obstacles:
                tmp_mgr = fcl.DynamicAABBTreeCollisionManager()
                tmp_mgr.registerObjects(obstacle['fcl_objs'])
                fcl_mgrs.append(tmp_mgr)
                mgrs_type.append('obstacle')
                
            min_distance = 1e10
            for i_idx in range(len(fcl_mgrs)):
                for j_idx in range(i_idx+1, len(fcl_mgrs)):
                    if not (mgrs_type[i_idx] == 'obstacle' and mgrs_type[j_idx] == 'obstacle'):
                    
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
            mgrs_type = []
            start_idx = 0
            for r_idx in range(self.n_robot):
                tmp_mgr = hppfcl.DynamicAABBTreeCollisionManager()
                tmp_objs = self.robots[r_idx].hppfcl_objs(jointPos=self.q[start_idx:start_idx+self.robots[r_idx].n_dof])
                for obj in tmp_objs:
                    tmp_mgr.registerObject(obj)
                hppfcl_mgrs.append(tmp_mgr)
                mgrs_type.append('robot')
                start_idx += self.robots[r_idx].n_dof
                
            for obstacle in self.obstacles:
                tmp_mgr = hppfcl.DynamicAABBTreeCollisionManager()
                tmp_objs = obstacle['hppfcl_objs']
                for obj in tmp_objs:
                    tmp_mgr.registerObject(obj)
                hppfcl_mgrs.append(tmp_mgr)
                mgrs_type.append('obstacle')

            min_distance = 1e10
            for i_idx in range(len(hppfcl_mgrs)):
                for j_idx in range(i_idx+1, len(hppfcl_mgrs)):
                    if not (mgrs_type[i_idx] == 'obstacle' and mgrs_type[j_idx] == 'obstacle'):
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
    
    def calculate_distance_between_objects(self, X, collision_pairs, pbar=False):
        if isinstance(X, torch.Tensor):
            output = torch.zeros(len(X), len(collision_pairs), 1).to(X).type(torch.float)
            X = X.detach().cpu().numpy()
        elif isinstance(X, np.ndarray):
            output = np.zeros((len(X), len(collision_pairs), 1))
            
        for b_idx in trange(len(X), disable=not pbar, desc='Min.dist', ncols=100):
            self.set_inputs(X[b_idx])
            for ls_idx, objects in enumerate(collision_pairs):
                output[b_idx, ls_idx] = self.env_bullet.get_distance_between_objects(objects[0], objects[1])
                
        return output
    
    def plot(self):
        img = self.get_image(width=1280, height=720, yaw=45)
        fig = go.Figure(go.Image(z=img)).update_layout(**plotly_layout, width=1280, height=720)
        return fig
    
    