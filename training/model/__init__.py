import numpy as np
import torch
import torch.functional as F
import sys, os

from training.model.PairwiseNet import PairwiseNet

def get_model(model_cfg, **kwargs):
    name = model_cfg['name']
    model = _get_model_instance(name)
    model = model(model_cfg, **kwargs)
    return model

def _get_model_instance(name):
    try:
        return {
            'PairwiseNet': get_PairwiseNet,
        }[name]
    except:
        raise (f'Model {name} not available.')

def get_PairwiseNet(model_dict, **kwargs):
    encoder_cfg = model_dict.encoder
    encoder = _get_model_instance(encoder_cfg.name)
    encoder = encoder(encoder_cfg, **kwargs)
    
    model = PairwiseNet(encoder=encoder, 
                        hidden_nodes=model_dict.hidden_nodes,
                        activation=model_dict.activation,
                        last_activation=model_dict.last_activation,
                        output_dims=model_dict.output_dims,
                        **kwargs)
    return model