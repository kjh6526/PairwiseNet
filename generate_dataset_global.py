import numpy as np
import torch
import os
import argparse
from omegaconf import OmegaConf

import pybullet as p
import open3d as o3d

from datetime import datetime

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
plotly_layout = dict(margin=dict(l=20, r=20, t=20, b=20))

from envs import get_env
from utils import save_yaml

def parse_arg_type(val):
    if val.isnumeric():
        return int(val)
    if (val == 'True') or (val == 'true'):
        return True
    if (val == 'False') or (val == 'false'):
        return False
    try:
        return float(val)
    except:
        return str(val)

def parse_unknown_args(l_args):
    """convert the list of unknown args into dict
    this does similar stuff to OmegaConf.from_cli()
    I may have invented the wheel again..."""
    n_args = len(l_args) // 2
    kwargs = {}
    for i_args in range(n_args):
        key = l_args[i_args*2]
        val = l_args[i_args*2 + 1]
        assert '=' not in key, 'optional arguments should be separated by space'
        kwargs[key.strip('-')] = parse_arg_type(val)
    return kwargs

def parse_nested_args(d_cmd_cfg):
    """produce a nested dictionary by parsing dot-separated keys
    e.g. {key1.key2 : 1}  --> {key1: {key2: 1}}"""
    d_new_cfg = {}
    for key, val in d_cmd_cfg.items():
        l_key = key.split('.')
        d = d_new_cfg
        for i_key, each_key in enumerate(l_key):
            if i_key == len(l_key) - 1:
                d[each_key] = val
            else:
                if each_key not in d:
                    d[each_key] = {}
                d = d[each_key]
    return d_new_cfg
        
def run(env, n_data, n_pcd, PATH):
    q_min = torch.as_tensor(env.q_min).squeeze()
    q_max = torch.as_tensor(env.q_max).squeeze()
    data_q = torch.rand(n_data, env.n_dof) * (q_max-q_min).repeat(n_data, 1) + q_min.repeat(n_data, 1)
    label = env.calculate_min_distance(data_q, pbar=True)
    
    torch.save(data_q, os.path.join(PATH, 'data_q.pt'))
    torch.save(label, os.path.join(PATH, 'label.pt'))
    
    mesh_pcd_path = os.path.join(PATH, 'pcds')
    os.makedirs(mesh_pcd_path)
    for o_idx in range(env.n_objects):
        bID, lID = env.env_bullet.idx2id(o_idx)
        linkinfo = p.getVisualShapeData(bID)[lID+1]
        pcd_path = os.path.join(os.path.dirname(linkinfo[4].decode('ascii')), f'pcd_{n_pcd}')
        pcd_file = os.path.basename(linkinfo[4].decode('ascii')).split('.')[0]+'.pt'
        pcd_file = os.path.join(pcd_path, pcd_file)
        if os.path.exists(pcd_file):
            pcd_object = torch.load(pcd_file)
        else:
            print(f'{pcd_file.split("/")[-1]} not found. Generating...')
            mesh = o3d.io.read_triangle_mesh(linkinfo[4])
            mesh.compute_vertex_normals()
            pcd_object = mesh.sample_points_uniformly(number_of_points=n_pcd)
            pcd_object = torch.tensor(np.array(pcd_object.points), dtype=torch.float).T
            os.makedirs(pcd_path, exist_ok=True)
            torch.save(pcd_object, pcd_file)
        
        torch.save(pcd_object, os.path.join(mesh_pcd_path, f'pcd_{o_idx}.pt'))
        
    print(f'{n_data} data points (joint configurations and the global collision distances) are successfully saved at {PATH}.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--n_data", type=int, default=10000)
    parser.add_argument("--n_pcd", type=int, default=100)
    args, unknown = parser.parse_known_args()
    d_cmd_cfg = parse_unknown_args(unknown)
    d_cmd_cfg = parse_nested_args(d_cmd_cfg)
    print(d_cmd_cfg)
    
    n_data = args.n_data
    n_pcd = args.n_pcd
    cfg = OmegaConf.load(args.env)
    cfg = OmegaConf.merge(cfg, d_cmd_cfg)
    print(OmegaConf.to_yaml(cfg))
    
    env = get_env(cfg, verbose=True)
    
    env_fig = env.plot()
    
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f'{run_id}_global_{n_data}'
    
    dataset_path = os.path.join('datasets', cfg.id, run_id)
    os.makedirs(dataset_path, exist_ok=False)
    save_yaml(os.path.join(dataset_path, 'env_config.yml'), OmegaConf.to_yaml(cfg))
    env_fig.write_image(os.path.join(dataset_path, 'env.png'))

    run(env, n_data, n_pcd, dataset_path)