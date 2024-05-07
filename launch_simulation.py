import pybullet as p
import torch
import numpy as np
import time, os, sys
import argparse
from omegaconf import OmegaConf

from envs import get_env

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str)
args, unknown = parser.parse_known_args()

env_cfg = OmegaConf.load(args.env)
env = get_env(env_cfg, debug=True, GUI=True)

select_pair_sd = p.addUserDebugParameter("# of pairs", 0, len(env.collision_pairs)-1, 0)
update_color_bt = p.addUserDebugParameter("Update color", 0, -1, 0)

update_flag_old = p.readUserDebugParameter(update_color_bt)

col_pair_old = None # [b1ID, l1ID, b2ID, l2ID]
col_pair_rgbcolor_old = None # [[rgba of obj 1], [rgba of obj 2]]

while True:
    pos = env.env_bullet.getTargetPositionfromGUI()
    env.env_bullet.reset2TargetPositions(pos)
    t = time.time()
    dist, points = env.env_bullet.check_collision(pos, return_points=True)
    t = time.time() - t
    print(f'Min. dist. {dist:.3f}m, elapsed time: {t:.3f}s')
    p.addUserDebugLine(points[0], points[1], [1, 0, 0], 5, 0.1)
    
    pair_idx = int(p.readUserDebugParameter(select_pair_sd))
    
    update_flag = p.readUserDebugParameter(update_color_bt)
    if update_flag != update_flag_old:
        update_flag_old = update_flag
        
        print(f'Change color of {pair_idx} pair.')
        
        if col_pair_old is not None:
            p.changeVisualShape(col_pair_old[0], col_pair_old[1], rgbaColor=col_pair_rgbcolor_old[0])
            p.changeVisualShape(col_pair_old[2], col_pair_old[3], rgbaColor=col_pair_rgbcolor_old[1])
        
        robot_color = [1.0, 0.0, 0.0, 1]
        b1ID, l1ID = env.env_bullet.idx2id(env.collision_pairs[pair_idx][0])
        b2ID, l2ID = env.env_bullet.idx2id(env.collision_pairs[pair_idx][1])
    
        color1_old = p.getVisualShapeData(b1ID)[l1ID+1][7]
        color2_old = p.getVisualShapeData(b2ID)[l2ID+1][7]
    
        p.changeVisualShape(b1ID, l1ID, rgbaColor=robot_color)
        p.changeVisualShape(b2ID, l2ID, rgbaColor=robot_color)
        
        col_pair_old = [b1ID, l1ID, b2ID, l2ID]
        col_pair_rgbcolor_old = [color1_old, color2_old]
        