import torch
import numpy as np
from torch.utils import data

from training.loader.global_dataset import Global
from training.loader.pairwise_dataset import Pairwise

def get_dataloader(data_dict, **kwargs):
    dataset = get_dataset(data_dict, **kwargs)
    loader = data.DataLoader(
        dataset,
        batch_size=data_dict["batch_size"],
        shuffle=data_dict.get("shuffle", True),
        num_workers=data_dict.get('n_workers', 0)
    )
    return loader

def get_dataset(data_dict, **kwargs):
    name = data_dict["dataset"]
    if name == 'global':
        dataset = Global(**data_dict, **kwargs)
    elif name == 'pairwise':
        dataset = Pairwise(**data_dict, **kwargs)
    else:
        raise NotImplementedError
    
    return dataset
        
class DEMO(torch.utils.data.Dataset):
    def __init__(self, in_dim, out_dim, n_data, **kwargs):
        self.data = torch.randn(n_data, in_dim)
        self.label = torch.randn(n_data, out_dim)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        return x, y