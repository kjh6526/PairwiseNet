import pybullet as p
import torch
import numpy as np
import time, os, sys
import argparse
from omegaconf import OmegaConf

from envs import get_env

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str)
parser.add_argument("--traj", type=str)
args, unknown = parser.parse_known_args()

env_cfg = OmegaConf.load(args.env)
env = get_env(env_cfg, GUI=True)

traj = torch.load(args.traj)
T = torch.linspace(0, 5, len(traj))

speed_slider = p.addUserDebugParameter(f'_Speed',rangeMin=0.01, rangeMax=3, startValue=1)

vStart = 0
bStart = p.addUserDebugParameter("_Start", 1, 0, vStart)

vReset = 0
bReset = p.addUserDebugParameter("_Reset", 1, 0, vReset)

vPause = 0
bPause = p.addUserDebugParameter("_Pause", 1, 0, vPause)

traj_idx = 0
env.env_bullet.reset2TargetPositions(traj[traj_idx])

ongoing_flag = False

t_old = time.time()
current_t = 0

while True:
    
    if int(p.readUserDebugParameter(bStart)) != vStart:
        vStart = int(p.readUserDebugParameter(bStart))
        print('Simulation Start')
        ongoing_flag = True
        
    if int(p.readUserDebugParameter(bReset)) != vReset:
        vReset = int(p.readUserDebugParameter(bReset))
        print('Simulation Reset')
        ongoing_flag = False
        env.env_bullet.reset2TargetPositions(traj[0])
        t_old = time.time()
        current_t = 0
        
    if int(p.readUserDebugParameter(bPause)) != vPause:
        vPause = int(p.readUserDebugParameter(bPause))
        print('Simulation Pause')
        ongoing_flag = False
    
    speed = p.readUserDebugParameter(speed_slider)    
    
    if ongoing_flag:
        current_t += (time.time() - t_old) * speed    
    t_old = time.time()

    traj_idx = abs(T-current_t).argmin()
        
    env.env_bullet.reset2TargetPositions(traj[int(traj_idx)])
    
    dist, points = env.env_bullet.check_collision(traj[int(traj_idx)], return_points=True)
    # print(f'Min. dist. {dist:.3f}m')
    p.addUserDebugLine(points[0], points[1], [1, 0, 0], 5, 0.1)
        