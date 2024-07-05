import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import AUROC
get_auroc = AUROC(task='binary')

import os
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from PIL import Image

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.subplots as sp
plotly_layout = dict(margin=dict(l=20, r=20, t=20, b=20))

from training.model.activations import get_activation


class PcdEncoder(nn.Module):
    def __init__(self, 
                encoder,
                decoder,
                **kwargs):
        super(PcdEncoder, self).__init__() 
        self.encoder = encoder
        self.decoder = decoder
        
            
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        pcd_embed = self.encoder(x)
        pcd_reconstruct = self.decoder(pcd_embed)
               
        return pcd_reconstruct
        
    def get_device(self):
        return list(self.parameters())[0].device

    def save(self, path):
        torch.save({'state_dict': self.state_dict()}, path)
        
    def train_step(self, x, criterion, optimizer, **kwargs):
        optimizer.zero_grad()
        output = self(x)
        loss = criterion(output, x)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}, output
    
    def validation_step(self, x, criterion, **kwargs):
        output = self(x)
        loss = criterion(output, x)
        return {"loss": loss.item()}
        
    def visualization_step(self, x, recon_x, **kwargs):
        """
                converts 2 pcd (tensor) data to npy data 
        """
        x = x.cpu().detach().numpy()
        recon_x = recon_x.cpu().detach().numpy()
        return {
                'viz/GT@': x,
                'viz/recon@': recon_x,
        }
