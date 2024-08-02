import numpy as np
import torch
import os, json, shutil
import argparse
from omegaconf import OmegaConf
import warnings
warnings.simplefilter("ignore")

from datetime import datetime
import pybullet as p
import open3d as o3d

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
plotly_layout = dict(margin=dict(l=20, r=20, t=20, b=20))

from envs import get_env
from envs.models.panda.panda import Panda
from envs.lib.LieGroup import invSE3
from utils import progress_tracker

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
        
def run(cfg, PATH):
    
    n_data = cfg.n_data
    n_pcd = cfg.n_pcd
    hand = cfg.hand

    n_radius = cfg.n_radius
    radius_range = cfg.radius_range
    
    n_theta = cfg.n_theta
    theta_range = cfg.theta_range
    
    n_phi = cfg.n_phi
    phi_range = cfg.phi_range
    
    n_env = n_radius*n_theta*n_phi
    
    n_data_per_env = int(np.ceil(n_data / n_env))

    pair_meshes_list = [None]*n_env
    T_12_list = [None]*n_env
    distances_list = [None]*n_env
    
    env_idx = 0
    
    pbar = progress_tracker(total=n_env, desc='n_env', ncols=100)
    
    # Mid  : "Unique" Mesh ID, 0 ~ len(np.unique(np.array(object_mesh_files)))-1
    # mesh : Mesh Name, ex) 'link0.obj'
    # one variable for all environments
    mesh2Mid_dict = {}
    
    for radius in np.linspace(*radius_range, n_radius):
        for theta in np.linspace(*theta_range, n_theta):
            for phi in np.linspace(*phi_range, n_phi):
                
                env_cfg = {
                    'name': 'multipanda',
                    'base_poses': [
                        [0, 0, 0],
                        [radius*np.cos(theta), radius*np.sin(theta), 0]
                    ],
                    'base_orientations': [0, phi],
                    'hand': hand,
                    'collision_pairs_config': {
                        'ignore': True,
                    }
                }
                
                env = get_env(env_cfg=env_cfg)
                
                # collision_pairs = torch.tensor(env.collision_pairs)
                collision_pairs = []
                for o1_idx in range(env.n_objects):
                    for o2_idx in [x for x in range(env.n_objects) if x != o1_idx]:
                        collision_pairs.append([o1_idx, o2_idx])
                        
                collision_pairs = torch.tensor(collision_pairs, dtype=torch.int64)
                n_collision_pairs = len(collision_pairs)
                
                object_mesh_files = []
                for o_idx in range(env.n_objects):
                    bID, lID = env.env_bullet.idx2id(o_idx)
                    linkinfo = p.getVisualShapeData(bID)[lID+1]
                    meshfile = linkinfo[4].decode('ascii')
                    object_mesh_files.append(meshfile)
                    if meshfile not in mesh2Mid_dict:
                        mesh2Mid_dict[meshfile] = len(mesh2Mid_dict)
                        
                # Oid  : Object ID, 0 ~ env.n_objects-1
                # one variable per each environment
                Oid2mesh_dict = dict(zip(np.arange(len(object_mesh_files)), object_mesh_files))    
                Oid2mesh_map = np.vectorize(Oid2mesh_dict.get)
                
                n_sample = int(np.ceil(n_data_per_env / n_collision_pairs))
                pair_indices = torch.stack([collision_pairs]*n_sample, dim=0) 
                
                q_min = torch.as_tensor(env.q_min).squeeze()
                q_max = torch.as_tensor(env.q_max).squeeze()
                data_q = torch.rand(n_sample, env.n_dof) * (q_max-q_min).repeat(n_sample, 1) + q_min.repeat(n_sample, 1)
                
                distances = env.calculate_distance_between_objects(data_q, collision_pairs, pbar=False)
                
                data_SE3 = env.get_Ts_objects(data_q)

                T_1 = data_SE3[:, pair_indices[0, :, 0]]
                T_2 = data_SE3[:, pair_indices[0, :, 1]]
                
                distances = distances.view(n_sample*n_collision_pairs, 1)[:n_data_per_env]
                pair_indices = pair_indices.view(n_sample*n_collision_pairs, 2)[:n_data_per_env]
                T_1 = T_1.view(n_sample*n_collision_pairs, 4, 4)[:n_data_per_env]
                T_2 = T_2.view(n_sample*n_collision_pairs, 4, 4)[:n_data_per_env]
                
                T_1_inv = invSE3(T_1)
                T_12 = torch.einsum('bij, bjk -> bik', T_1_inv, T_2)
                
                pair_meshes = Oid2mesh_map(pair_indices.cpu().numpy())

                pair_meshes_list[env_idx] = pair_meshes
                T_12_list[env_idx] = T_12
                distances_list[env_idx] = distances
                
                env_idx += 1
                pbar.update(env_idx)
    
    pbar.close()
    
    distances = torch.cat(distances_list, dim=0)
    T_12 = torch.cat(T_12_list, dim=0)
    
    pair_meshes = np.concatenate(pair_meshes_list, axis=0)
    mesh2Mid_map = np.vectorize(mesh2Mid_dict.get)
    pair_indices = torch.tensor(mesh2Mid_map(pair_meshes), dtype=torch.int64, device=distances.device)
    
    # dictionary from Mid to mesh name
    Mid2mesh_dict = {}
    mesh_data_path = os.path.join(PATH, 'mesh_data')
    os.makedirs(mesh_data_path)
    for meshfile, Mid in mesh2Mid_dict.items():
        meshname = os.path.basename(meshfile)
        shutil.copy(meshfile, os.path.join(mesh_data_path, meshname))
        Mid2mesh_dict[Mid] = meshname
        
        # Save pcd files for each mesh (for fixed pcds)
        pcd_path = os.path.join(os.path.dirname(meshfile), f'pcd_{n_pcd}')
        pcd_file = os.path.basename(meshfile).split('.')[0]+'.pt'
        pcd_file = os.path.join(pcd_path, pcd_file)
        
        if os.path.exists(pcd_file):
            pcd_object = torch.load(pcd_file)
        else:
            print(f'{pcd_file.split("/")[-1]} not found. Generating...')
            mesh = o3d.io.read_triangle_mesh(meshfile)
            pcd_object = mesh.sample_points_uniformly(number_of_points=n_pcd)
            pcd_object = torch.tensor(np.array(pcd_object.points), dtype=torch.float).T
            os.makedirs(pcd_path, exist_ok=True)
            torch.save(pcd_object, pcd_file)
            
        torch.save(pcd_object, os.path.join(mesh_data_path, f'{meshname}_pcd_{n_pcd}.pt'))
        
    print(f'distance : {distances.shape}')
    print(f'pair_indices : {pair_indices.shape}')
    print(f'T_12 : {T_12.shape}')
    print(f'Mid2mesh_dict : {Mid2mesh_dict}')
    
    torch.save(distances, os.path.join(PATH, 'distances.pt'))
    torch.save(pair_indices, os.path.join(PATH, 'pair_indices.pt'))
    torch.save(T_12, os.path.join(PATH, 'T_12.pt'))
    with open(os.path.join(PATH, 'Mid2mesh_dict.json'), 'w') as dict_file:
        json.dump(Mid2mesh_dict, dict_file)
        
    print(f'{n_data} points of pairwise collision distance data are successfully saved at {PATH}.')
    print(f'Total {n_env} environments, {n_data_per_env} data points per environment.')
    print(f'{n_sample} joint configurations are used per environment on average.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args, unknown = parser.parse_known_args()
    d_cmd_cfg = parse_unknown_args(unknown)
    d_cmd_cfg = parse_nested_args(d_cmd_cfg)
    
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, d_cmd_cfg)
    print(cfg)
    
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    if cfg.hand:
        run_id = f'{run_id}_hand'
    run_id = f'{run_id}_{cfg.n_data}_{cfg.n_pcd}'
    
    dataset_path = os.path.join('datasets', 'Multiarm_Pairwise', run_id)
    os.makedirs(dataset_path, exist_ok=False)

    run(cfg, dataset_path)