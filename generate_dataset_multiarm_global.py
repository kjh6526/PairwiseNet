import numpy as np
import torch
import os
import argparse
from omegaconf import OmegaConf

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
        
def run(env, n_data, PATH):
    q_min = torch.as_tensor(env.q_min).squeeze()
    q_max = torch.as_tensor(env.q_max).squeeze()
    data_q = torch.rand(n_data, env.n_dof) * (q_max-q_min).repeat(n_data, 1) + q_min.repeat(n_data, 1)
    label = env.calculate_min_distance(data_q, pbar=True)
    
    torch.save(data_q, os.path.join(PATH, 'data_q.pt'))
    torch.save(label, os.path.join(PATH, 'label.pt'))
        
    print(f'{n_data} data points (joint configurations and the global collision distances) are successfully saved at {PATH}.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--n_data", type=int, default=10000)
    args, unknown = parser.parse_known_args()
    d_cmd_cfg = parse_unknown_args(unknown)
    d_cmd_cfg = parse_nested_args(d_cmd_cfg)
    print(d_cmd_cfg)
    
    n_data = args.n_data
    cfg = OmegaConf.load(args.env)
    cfg = OmegaConf.merge(cfg, d_cmd_cfg)
    print(OmegaConf.to_yaml(cfg))
    
    env = get_env(cfg)
    
    env_fig = env.plot()
    
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_id = f'{run_id}_{env.n_dof}DOF_{n_data}'
    
    dataset_path = os.path.join('datasets', cfg.name, run_id)
    os.makedirs(dataset_path, exist_ok=False)
    save_yaml(os.path.join(dataset_path, 'env_config.yml'), OmegaConf.to_yaml(cfg))
    env_fig.write_image(os.path.join(dataset_path, 'env.png'))

    run(env, n_data, dataset_path)