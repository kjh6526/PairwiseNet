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

while True:
    pos = env.env_bullet.getTargetPositionfromGUI()
    env.env_bullet.reset2TargetPositions(pos)
    t = time.time()
    dist, points = env.env_bullet.check_collision(pos, return_points=True)
    t = time.time() - t
    print(f'Min. dist. {dist:.3f}m, elapsed time: {t:.3f}s')
    p.addUserDebugLine(points[0], points[1], [1, 0, 0], 5, 0.1)