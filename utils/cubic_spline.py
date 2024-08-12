import torch
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
from torch.nn.parameter import Parameter

class cubic_spline_curve(torch.nn.Module):
    def __init__(self, z_i, z_f, device, lengths=2):
        super(cubic_spline_curve, self).__init__()
        self.z_i = z_i.unsqueeze(0)
        self.z_f = z_f.unsqueeze(0)
        self.device = device
        self.z = Parameter(
                    torch.cat(
                        [self.z_i + (self.z_f-self.z_i) * t / (lengths + 1) + torch.randn_like(self.z_i)*0.0 for t in range(1, lengths+1)], dim=0)
        )
        self.t_linspace = torch.linspace(0, 1, lengths + 2).to(self.device)

    def append(self):
        return torch.cat([self.z_i, self.z, self.z_f], dim=0)
    
    def spline_gen(self):
        coeffs = natural_cubic_spline_coeffs(self.t_linspace, self.append())
        spline = NaturalCubicSpline(coeffs)
        return spline
    
    def forward(self, t):
        out = self.spline_gen().evaluate(t)
        return out
    
    def velocity(self, t):
        out = self.spline_gen().derivative(t)
        return out

    def length(self, func=None, num_discretizations=100):
        t_samples = torch.linspace(0, 1, num_discretizations).to(self.device)
        z_samples = self(t_samples)
        if func is not None:
            z_samples = func(z_samples).reshape(num_discretizations, -1)
        delta_z_samples = z_samples[1:] - z_samples[:-1]
        return torch.einsum('ni,ni->n', delta_z_samples, delta_z_samples).sum()
    
class cubic_spline_curve_manual(torch.nn.Module):
    def __init__(self, path, device):
        super(cubic_spline_curve_manual, self).__init__()
        self.dim = path.shape[1]
        lengths = len(path) - 2
        
        self.z_i = path[0].unsqueeze(0)
        self.z_f = path[-1].unsqueeze(0)
        self.device = device
        
        self.z = Parameter(path[1:-1])
        self.t_linspace = torch.linspace(0, 1, lengths + 2).to(self.device)

    def append(self):
        return torch.cat([self.z_i, self.z, self.z_f], dim=0)
    
    def spline_gen(self):
        coeffs = natural_cubic_spline_coeffs(self.t_linspace, self.append())
        spline = NaturalCubicSpline(coeffs)
        return spline
    
    def forward(self, t):
        out = self.spline_gen().evaluate(t)
        return out
    
    def velocity(self, t):
        out = self.spline_gen().derivative(t)
        return out

    def length(self, func=None, num_discretizations=10000):
        t_samples = torch.linspace(0, 1, num_discretizations).to(self.device)
        z_samples = self(t_samples)
        if func is not None:
            z_samples = func(z_samples).reshape(num_discretizations, -1)
        delta_z_samples = z_samples[1:] - z_samples[:-1]
        return torch.einsum('ni,ni->n', delta_z_samples, delta_z_samples).sqrt().sum()