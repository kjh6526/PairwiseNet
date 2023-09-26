import torch
import torch.functional as F

def get_activation(**kwargs):
    name = kwargs['name']
    if name == 'relu':
        activation = torch.relu
    elif name == 'tanh':
        activation = torch.nn.Tanh()
    elif name == 'linear':
        activation = lambda x: x
    elif name == 'sigmoid':
        activation = torch.sigmoid
    elif name == 'softmax':
        dim = kwargs.get('dim', -1)
        activation = torch.nn.Softmax(dim=dim)
    elif name == 'leakyrelu':
        activation = torch.nn.LeakyReLU()
    else:
        raise NotImplementedError
    return activation