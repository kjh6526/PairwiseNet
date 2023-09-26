import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import torch
from tqdm import tqdm, trange

from envs.multipanda import MultiPanda

def get_env(env_cfg, **kwargs):
    name = env_cfg['name']

    if name == 'multipanda':
        env = MultiPanda(**env_cfg, **kwargs)
    else:
        raise NotImplementedError
    
    return env

