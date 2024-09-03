## CLIENT ##
# -*- coding: utf-8 -*- 

import socket
from _thread import *
import numpy as np
import time
import ast
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

import os
import pickle
import csv

import sys
from scipy.spatial.transform import Rotation as Rot
from training.model import get_model
import yaml
import argparse
from envs import get_env
from omegaconf import OmegaConf
from envs.lib.LieGroup import invSE3

from datetime import datetime

def now():
    return datetime.now().strftime('%y%m%d%H%M')

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
print(device)


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
   
# define listener
HOST = '***.***.***.***' # '127.0.1.1' ## server에 출력되는 ip를 입력해주세요 ##
PORT = 5555
fps = 1e8
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# client_socket.connect((HOST, PORT))
class Listener:
    def __init__(self) -> None:
        self.msg_s2c = 'None'
    def recv_data(self, client_socket):
        while True:
            data = client_socket.recv(1024)
            # self.msg_s2c = client_socket.recv(1024)
            dd = data.decode()
            idx_e = dd.rfind(']')
            idx_s = dd[:idx_e].rfind('[')
            if idx_e == -1 or idx_s == -1:
                # print(idx_e, idx_s)
                # print(dd)
                # print('index not found...')
                # time.sleep(0.1)
                pass
            else:
                self.msg_s2c = dd
                self.msg_s2c = self.msg_s2c[idx_s: idx_e+1]
                # msg.str = dd[idx_s: idx_e+1]
            # print("recive : ", repr(self.msg_s2c))
    client_socket.connect((HOST, PORT))
    
def send_array_msg_c2s(array, client_socket=client_socket):
    msg_c2s = str(array.tolist())
    client_socket.send(msg_c2s.encode())


    
lis = Listener()
start_new_thread(lis.recv_data, (client_socket,))
print('>> Connect Server')


ts = time.time()
tss = time.time()
i = 0

path = 'results/20240717-140346'
cfg_path =  path + '/config_PairwiseNet_labpanda.yml'
device = 0

cfg = OmegaConf.load(cfg_path)
env_cfg = OmegaConf.load(os.path.join(cfg.data.test.root, 'env_config.yml'))


env = get_env(env_cfg, device=device, GUI = False)

time_step = 0

d_min_old = 0.38618

model = get_model(cfg.model, env=env).to(device)
checkpoint = torch.load(path+'/model_best_accuracy.pkl')
model_state_dict = checkpoint['model_state']
model.load_state_dict(model_state_dict)   
pairwise = Pairwise2Global(model, cfg, env) 

# constant values used for control
vel_spring = 40
vel_damper = 150
intensity_amplify_ratio = 200
d_nice = 0.05
repulsion_intensity = 1.0

with open(f'results/log_data_{now()}.csv', mode='a', newline='') as file:
    writer = csv.writer(file)

    if file.tell() == 0:
        writer.writerow(['time_step', 'time', 'target_vel', 'intensity', 'current_joint', 'min_distance'])

    while True:
        time_start = time.time()
        
        if lis.msg_s2c == 'None':
            continue
        
        current_joint = ast.literal_eval(lis.msg_s2c)        
        current_joint = torch.tensor(current_joint, dtype = torch.float, device = device, requires_grad = True)
        input_vec = current_joint.unsqueeze(0) 
        d_min = pairwise(input_vec).squeeze()
        d_min.backward()
        gradient_dmin = current_joint.grad

        if d_min > d_nice:
            target_vel = torch.zeros_like(current_joint)
            repulsion_intensity = 1.0

        else:
            target_vel = gradient_dmin / torch.norm(gradient_dmin) * (vel_spring * (d_nice-d_min) - vel_damper*(d_min - d_min_old)) 
            repulsion_intensity = 1.0 + intensity_amplify_ratio*(d_nice-d_min)

        joint_max = torch.tensor([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype = torch.float, device = device)
        target_vel = torch.clamp(target_vel, min=-joint_max, max=joint_max)

        repulsion_intensity = torch.tensor([repulsion_intensity], dtype = torch.float, device = device)
        repulse_vec = torch.cat((target_vel, repulsion_intensity))
        # send_array_msg_c2s(torch.zeros_like(repulse_vec))
        send_array_msg_c2s(repulse_vec)
        d_min_old = d_min
        i+=1
        print(f'\nrepulse_vec: {repulse_vec}, dmin: {d_min}')
        # time.sleep(1)
        # print('msg = ' + str(array))
        if i % 100000 == 0:
            i = 0
            print(f'\nTime elapsed: {time.time() - ts}, FPS: {100000 / (time.time() - ts)}, msg_from_r: {lis.msg_s2c}')
            ts = time.time()
            # time.sleep(2)
        # toc = time.time()
        # print((toc - tic), 1/fps, 1/fps - (toc - tic))
        # if toc - tic < 1/fps:
        #     time.sleep(1/fps - (toc - tic))
        
        #will log data only when collision repulsion occured!!
    
        time_log = time.time()
        target_vel = target_vel.cpu().tolist()
        repulsion_intensity = repulsion_intensity.cpu().item()
        current_joint = current_joint.cpu().tolist()
        d_min = d_min.cpu().item()
        writer.writerow([i, time_log, target_vel, repulsion_intensity, current_joint, d_min])
    
        time_end = time.time()    
        print(f'FPS: {1/(time_end - time_start)}')

client_socket.close()