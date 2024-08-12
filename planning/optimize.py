import torch

from tqdm import tqdm, trange
import copy
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.subplots as sp
plotly_layout = dict(margin=dict(l=20, r=20, t=20, b=20))

from utils import now
from utils.cubic_spline import cubic_spline_curve, cubic_spline_curve_manual

class TrajectoryOptimizer:
    def __init__(self, to_cfg):
        self.length = to_cfg.get('length', 'joint')
        self.num_sample = to_cfg.get('num_sample', 1000)
        self.mu_g = to_cfg.get('mu_g', 10)
        self.mu_v = to_cfg.get('mu_v', 0.1)
        
        col_thr = to_cfg.get('col_thr', (0.01, 'lower'))
        
        self.col_thr = col_thr[0]
        self.col_ineq = col_thr[1]
        
    def optimize(self, curve, col_model, device, env, iteration=1000, pbar=False, verbose=True, **kwargs):
        
        results = {}
        results['f_hist'] = []
        results['g_hist'] = []
        results['dX_hist'] = []
        results['loss_hist'] = []
        
        plotly_figwidget = kwargs.get('figwidget', None)
        
        num_samples = self.num_sample
        mu_g = torch.ones(num_samples).to(device) * self.mu_g
        mu_v = torch.ones(1).to(device) * self.mu_v
        
        col_thr = self.col_thr
        
        opt = torch.optim.Adam(curve.parameters(), lr=0.01)
        
        if verbose:
            print(now() + ' Trajectory Optimization Start.')
        
        min_loss = 1e10
        best_curve = None
        
        for _ in trange(iteration, disable=not pbar, desc='Opt.', ncols=100):
            opt.zero_grad()
            if self.length == 'joint':
                f = curve.length()
            elif self.length == 'cartesian':
                f = curve.length(func=env.get_Ps)
            elif self.length == 'joint+cartesian':
                f = curve.length() + curve.length(func=env.get_Ps)
            # f = curve.velocity(torch.rand(num_samples)).norm(dim=1).mean()
            
            sampled_ts = torch.rand(num_samples).to(curve.device)
            sample_points = curve(sampled_ts)
            min_dist = col_model(sample_points).squeeze()
            if self.col_ineq == 'lower':
                g = mu_g @ torch.clamp(col_thr-min_dist, min=0)
            elif self.col_ineq == 'upper':
                g = mu_g @ torch.clamp(min_dist-col_thr, min=0)
            
            lin_ts = torch.linspace(0, 1, num_samples).to(curve.device)
            lin_points = curve(lin_ts)
            lin_Xs = env.get_Ps(lin_points)
            dX = (lin_Xs[1:] - lin_Xs[:-1]).norm(dim=2).max() * num_samples
            g_dX = mu_v * dX

            # loss = f + g
            # loss = f + g + g_dX
            loss = f + g + g_dX + ((min_dist - col_thr)**2).mean()
            
            loss.backward()
            opt.step()
            
            results['f_hist'].append(f.item())
            results['g_hist'].append(g.item())
            results['dX_hist'].append(g_dX.item())
            results['loss_hist'].append(loss.item())
            
            if plotly_figwidget:
                for plot in plotly_figwidget.data:
                    if plot.name == 'loss':
                        plot.y += (results['loss_hist'][-1], )
                    elif plot.name == 'g':
                        plot.y += (results['g_hist'][-1], )
                    elif plot.name == 'dX':
                        plot.y += (results['dX_hist'][-1], )
                    elif plot.name == 'f':
                        plot.y += (results['f_hist'][-1], )
            
            if min_loss > loss.item() and g.item() < 1e-5:
                min_loss = loss.item()
                best_curve = copy.deepcopy(curve)
                
        if verbose:
            print(now() + ' Trajectory Optimization End.')
            
        return results, curve, min_loss, best_curve