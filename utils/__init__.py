import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import time, yaml, os, copy

import open3d as o3d

# os.path.join('..')
# from training.model import get_model

class progress_tracker():
    def __init__(self, **kwargs):
        self.tqdm_obj = tqdm(**kwargs)
        self.value_old = 0
        
    def update(self, x):
        self.tqdm_obj.update(x - self.value_old)
        self.value_old = x
        self.tqdm_obj.refresh()
    
    def close(self):
        self.tqdm_obj.close()

class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def now():
    return f'[{time.strftime("%Y/%m/%d %X", time.localtime())}]'

def save_yaml(filename, text):
    """parse string as yaml then dump as a file"""
    with open(filename, "w") as f:
        yaml.dump(yaml.safe_load(text), f, default_flow_style=False)
        
def get_info_from_cfg(cfg, keys):
    """_summary_
    Args:
        cfg (Omegaconf Config): Config file
        keys (list of string): list of keys to extract from the config file

    Returns:
        dictionary: dictionary of extracted keys
    """
    output = {}
    for key in keys:
        if key == 'epoch':
            output[key] = cfg.training.n_epoch
        elif key == 'seed':
            output[key] = cfg.training.seed
        elif key == 'train_batch':
            output[key] = cfg.data.training.batch_size
        elif key == 'lr':
            output[key] = cfg.training.optimizer.lr
        elif key == 'train_dataset':
            output[key] = cfg.data.training.root
        else:
            raise ValueError(f'key {key} not found in the config file')
        
    return output

# def get_model_from_result_dir(result_dir, device='cpu', best_model_suffix='best.pkl'):
#     """_summary_
#     Args:
#         result_dir (str): path to the result directory

#     Returns:
#         model: torch model
#         cfg: OmegaConf Config
#     """
    
#     cfg, best_model = None, None
#     for file in os.listdir(result_dir):
#         if file.endswith('yml'):
#             cfg = OmegaConf.load(os.path.join(result_dir, file))
#         if file.endswith(best_model_suffix):
#             best_model = torch.load(os.path.join(result_dir, file))

#     assert cfg is not None, 'cfg file does not exist.'
#     assert best_model is not None, 'best_model does not exist.'

#     model = get_model(cfg.model).to(device)
#     model.load_state_dict(best_model['model_state'])
#     return model, cfg

def make_convex_mesh(mesh):
    """_summary_
    Args:
        mesh (o3d.geometry.TriangleMesh): input mesh

    Returns:
        o3d.geometry.TriangleMesh: convex hull of the input mesh
    """
    convex_mesh = copy.deepcopy(mesh)
    convex_mesh.remove_duplicated_vertices()
    convex_mesh.remove_duplicated_triangles()
    convex_mesh.remove_degenerate_triangles()
    convex_mesh.compute_convex_hull()
    convex_mesh.orient_triangles()
    return convex_mesh