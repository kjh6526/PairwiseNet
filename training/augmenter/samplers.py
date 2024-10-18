import torch
import numpy as np

from tqdm import tqdm, trange
    
class UniformSampler:
    def __init__(self, **kwargs):
        pass
        
    def sample(self, n, ub, lb, **kwargs):
        ub = torch.as_tensor(ub)
        lb = torch.as_tensor(lb)
        dim = len(ub)
        
        X = torch.rand(n, dim).to(ub)
        X = X * (ub-lb).repeat(n, 1) + lb.repeat(n, 1)    
        return X
    
class MCMCSampler:
    def __init__(self, **kwargs):
        self.sigma = kwargs.get('sigma', 0.1)
        self.step_sigma = kwargs.get('step_sigma', 1)
        self.iteration = kwargs.get('iteration', 1000)
        self.o_target = kwargs.get('output_target', 0.01)
        self.thr_lb = kwargs.get('thr_lb', 0.8)
    
    def sample(self, n, ub, lb, model, **kwargs):
        device = model.get_device()
        ub = torch.as_tensor(ub).to(device)
        lb = torch.as_tensor(lb).to(device)
        dim = len(ub)
        X = torch.rand(n, dim).to(device)
        X = X * (ub-lb).repeat(n, 1) + lb.repeat(n, 1)
        
        for i_iter in trange(self.iteration, disable=not kwargs.get('pbar', False), desc='MCMC', ncols=100):
            fx = model(X).squeeze()
            Ex = (fx-self.o_target)**2 / self.sigma**2
            Px = torch.exp(-Ex)
            
            # X_new = self.radian_clamp(X + torch.randn(X.shape).to(X) * self.step_sigma)
            X_new = self.clamping(X + torch.randn(X.shape).to(X) * self.step_sigma, ub, lb)
            fx_new = model(X_new).squeeze()
            Ex_new = (fx_new-self.o_target)**2 / self.sigma**2
            Px_new = torch.exp(-Ex_new)
            
            ratio = Px_new / (Px + 1e-10)
            thr = torch.FloatTensor(ratio.shape).uniform_(self.thr_lb, 1).to(ratio)
            X[ratio>thr, :] = X_new[ratio>thr, :]
            
        return X
    
    @staticmethod       
    def radian_clamp(x):
        while (x < -torch.pi).any() or (x >= torch.pi).any():
            x[x >= torch.pi] -= 2*torch.pi
            x[x < -torch.pi] += 2*torch.pi
        return x
    
    @staticmethod
    def clamping(x, ub, lb):
        dim = len(ub.squeeze())
        ub = ub.view(-1, dim)
        lb = lb.view(-1, dim)
        x = torch.max(torch.min(x, ub), lb)
        return x
    
class LangevinMCSampler:
    def __init__(self, **kwargs):
        self.sigma = kwargs.get('sigma', 0.3)
        self.step_sigma = kwargs.get('step_sigma', 0.05)
        self.iteration = kwargs.get('iteration', 1000)
        self.o_target = kwargs.get('output_target', 0.01)
        self.thr_lb = kwargs.get('thr_lb', 0.0)
        self.step = kwargs.get('step', 0.1)
    
    def sample(self, n, ub, lb, model, **kwargs):
        device = model.get_device()
        ub = torch.as_tensor(ub).to(device)
        lb = torch.as_tensor(lb).to(device)
        dim = len(ub)
        X = torch.rand(n, dim).to(device)
        X = X * (ub-lb).repeat(n, 1) + lb.repeat(n, 1)
        
        def Px_fn(X):
            fx = model(X).squeeze()
            # fx = model(X, return_pairwise=True).squeeze().mean(dim=1)
            Ex = (fx-self.o_target)**2 / self.sigma**2
            Px = torch.exp(-Ex)
            return Px
        
        X.requires_grad_()
        
        for i_iter in trange(self.iteration, disable=not kwargs.get('pbar', False), desc='LMC', ncols=100):
            Px = Px_fn(X)
            grad = torch.autograd.grad(Px, X, grad_outputs=torch.ones_like(Px))[0]
            
            X_new = self.clamping(X.detach() + self.step * grad + torch.randn(X.shape).to(X) * self.step_sigma, ub, lb)
            Px_new = Px_fn(X_new)
            
            ratio = Px_new / (Px + 1e-10)
            thr = torch.FloatTensor(ratio.shape).uniform_(self.thr_lb, 1).to(ratio)
            
            X = X.detach()
            X[ratio>thr, :] = X_new[ratio>thr, :]
            X.requires_grad_()
            
        X = X.detach()
        return X

    @staticmethod       
    def radian_clamp(x):
        while (x < -torch.pi).any() or (x >= torch.pi).any():
            x[x >= torch.pi] -= 2*torch.pi
            x[x < -torch.pi] += 2*torch.pi
        return x
    
    @staticmethod
    def clamping(x, ub, lb):
        dim = len(ub.squeeze())
        ub = ub.view(-1, dim)
        lb = lb.view(-1, dim)
        x = torch.max(torch.min(x, ub), lb)
        return x