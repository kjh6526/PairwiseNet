import numpy as np
import torch
import torch.functional as F
import sys, os

from training.model.dgcnn import DGCNN
from training.model.PairwiseNet import PairwiseNet


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
        }[name]
    except:
        raise (f'Model {name} not available.')


def get_DGCNN(model_dict, **kwargs):
    model = DGCNN(**model_dict, **kwargs)
    return model


def get_PairwiseNet(model_dict, **kwargs):
    encoder_cfg = model_dict.encoder
    encoder = _get_model_instance(encoder_cfg.name)
    use_pretrained_encoder = encoder_cfg.get('pretrain', False)
    if use_pretrained_encoder:
        # Load the pre-trained encoder from the specified file
        encoder = encoder(encoder_cfg, **kwargs)
        checkpoint = torch.load(encoder_cfg.root)
        model_state_dict = checkpoint['model_state']
        encoder_state_dict = {
            k.replace('encoder.', ''): v
            for k, v in model_state_dict.items()
            if k.startswith('encoder.')
        }
        encoder.load_state_dict(encoder_state_dict, strict=False)

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
