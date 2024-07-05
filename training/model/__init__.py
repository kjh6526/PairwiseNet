import numpy as np
import torch
import torch.functional as F
import sys, os

from training.model.dgcnn import DGCNN
from training.model.MLP_pcd import MLP_PointCloud
from training.model.PairwiseNet import PairwiseNet
from training.model.PcdEncoder import PcdEncoder

def get_model(model_cfg, **kwargs):
    name = model_cfg['name']
    model = _get_model_instance(name)
    model = model(model_cfg, **kwargs)
    return model

def _get_model_instance(name):
    try:
        return {
            'DGCNN': get_DGCNN,
            'PairwiseNet': get_PairwiseNet,
            'MLP_pcd': get_MLP_pcd,
            'PCD_autoencoder': get_PCD_autoencoder,
        }[name]
    except:
        raise (f'Model {name} not available.')

def get_DGCNN(model_dict, **kwargs):
    model = DGCNN(**model_dict, **kwargs)
    return model

def get_MLP_pcd(model_dict, **kwargs):
    model = MLP_PointCloud(**model_dict, **kwargs)
    return model

def get_PairwiseNet(model_dict, **kwargs):
    encoder_cfg = model_dict.encoder
    encoder = _get_model_instance(encoder_cfg.name)
    if encoder_cfg.pretrain:
        # Load the pre-trained encoder from the specified file
        encoder = encoder(encoder_cfg, **kwargs)
        pretrained_state = torch.load(encoder_cfg.root)
        encoder.load_state_dict(pretrained_state)
        
        # Set the encoder weights to not be updated during training
        if not encoder_cfg.finetune:
            for param in encoder.parameters():
                param.requires_grad = False
    else:
        encoder = encoder(encoder_cfg, **kwargs)
    model = PairwiseNet(encoder=encoder, 
                        hidden_nodes=model_dict.hidden_nodes,
                        activation=model_dict.activation,
                        last_activation=model_dict.last_activation,
                        output_dims=model_dict.output_dims,
                        **kwargs)
    return model

def get_PCD_autoencoder(model_dict, **kwargs):
    encoder_cfg = model_dict.encoder
    encoder = _get_model_instance(encoder_cfg.name)
    encoder = encoder(encoder_cfg, **kwargs)

    decoder_cfg = model_dict.decoder 
    decoder = _get_model_instance(decoder_cfg.name)
    decoder = decoder(decoder_cfg, **kwargs)

    model = PcdEncoder(encoder=encoder, 
                        decoder=decoder,
                        **kwargs)
    return model


    
