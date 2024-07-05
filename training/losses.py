import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_loss(cfg, **kwargs):
    name = cfg['name']
    if name == 'bce' or name == 'BCE':
        return torch.nn.BCELoss()
    elif name == 'mse' or name == 'MSE':
        return torch.nn.MSELoss()
    elif name == 'mspe' or name == 'MSPE':
        return MSPE(**cfg, **kwargs)
    elif name == 'chamfer' or name == 'CHAMFER':
        return ChamferLoss()
    else:
        raise NotImplementedError

def weighted_BCE_loss(output, target, weight=None):
    
    output = torch.clamp(output, min=1e-6, max=1-1e-6)
    
    if weight is None:
        weights = [1, 1]
    else:
        weights = [2-weight, weight]
        
    loss = weights[1] * (target * torch.log(output)) + \
           weights[0] * ((1 - target) * torch.log(1 - output))

    return torch.neg(torch.mean(loss))

def weighted_focal_loss(output, target, weight=None, gamma=2):
    
    output = torch.clamp(output, min=1e-6, max=1-1e-6)
    
    if weight is None:
        weights = [1, 1]
    else:
        weights = [2-weight, weight]
        
    loss = weights[1] * ((1-output)**gamma) * (target * torch.log(output)) + \
           weights[0] * (output**gamma) * ((1 - target) * torch.log(1 - output))

    return torch.neg(torch.mean(loss))

def weighted_BCE_with_logits_loss(output, target, weight=None):
    size = output.shape[0]
    
    if weight is None:
        weight = 1
        
    pos_weight = torch.tensor(np.ones((size, 1))*weight, dtype=torch.float32).to(output)
    criterion = torch.nn.BCEWithLogitsLoss(size_average=True, pos_weight=pos_weight)
    
    return criterion(output, target)

def MSELoss(output, target, weight=None):
    
    criterion = torch.nn.MSELoss()
    
    return criterion(output, target)

def ClsRegLoss(output, target, weight=1.0):
    cls_loss = weighted_BCE_loss(output[:, :-1], target[:, :-1])
    reg_loss = torch.nn.MSELoss()(output[:, -1], target[:, -1])
    
    return cls_loss + weight * reg_loss

class MSPE:
    def __init__(self, epsilon=1e-5, origin=0.0, eta=1.0, **kwargs):
        self.eps = epsilon
        self.origin = origin
        self.eta = eta
    def __call__(self, output, target):
        tmp = torch.clamp(torch.abs(target - self.origin), min=self.eps) ** (self.eta)
        loss = ((output - target) / tmp) ** 2
        return torch.mean(loss)

class ChamferLoss(nn.Module):
    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def batch_pairwise_dist(self, x, y):
        x = x.transpose(1,2)
        y = y.transpose(1,2)
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        diag_ind_x = torch.arange(0, num_points_x)
        diag_ind_y = torch.arange(0, num_points_y)
        if x.get_device() != -1:
            diag_ind_x = diag_ind_x.cuda(x.get_device())
            diag_ind_y = diag_ind_y.cuda(x.get_device())
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        loss_1 = torch.sum(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.sum(mins)
        return loss_1 + loss_2
		