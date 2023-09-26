#############################################
#                                           #
# code from IRVAE_public (Lee et al., 2022) #
#                                           #
#############################################

from cgi import test
import os, sys
import time
import math
from tkinter import Y

import numpy as np
import torch
import copy
from tqdm import tqdm, trange

sys.path.append('../..')
from utils import averageMeter, now
from training.losses import get_loss

class BaseTrainer:
    def __init__(self, optimizer, training_cfg, device, **kwargs):
        self.training_cfg = training_cfg
        self.device = device
        self.optimizer = optimizer
        self.criterion = get_loss(training_cfg['loss'])

    def train(self, model, d_dataloaders, logger=None, logdir="", **kwargs):
        cfg = self.training_cfg
    
        time_meter = averageMeter()
        train_loader, val_loader, test_loader = (d_dataloaders["training"], d_dataloaders["validation"], d_dataloaders["test"])
        i_iter = kwargs.get('iter_bias', 0)
        # best_val_loss = kwargs.get('best_val_loss', np.inf)
        best_val_loss = np.inf
    
        for i_epoch in range(1, cfg['n_epoch'] + 1):
            for x, y in train_loader:
                i_iter += 1

                model.train()
                start_ts = time.time()
                d_train = model.train_step(x.to(self.device), y.to(self.device), self.criterion, optimizer=self.optimizer, **kwargs)
                time_meter.update(time.time() - start_ts)
                logger.process_iter_train(d_train)

                if i_iter % cfg.print_interval == 0:
                    d_train = logger.summary_train(i_iter, reset=True)
                    print(
                        f"{now()} Epoch [{i_epoch:d}] Iter [{i_iter:d}]\tAvg Loss: {d_train['loss/train_loss_']:.6f}\tElapsed time: {time_meter.sum:.4f}\t"
                    )
                    time_meter.reset()
                    
                model.eval()
                if i_iter % cfg.val_interval == 0:
                    for x, y in tqdm(val_loader, disable=not cfg.get('pbar', False), ncols=100, desc='val'):
                        d_val = model.validation_step(x.to(self.device), y.to(self.device), self.criterion)
                        logger.process_iter_val(d_val)
                    d_val = logger.summary_val(i_iter, reset=True)
                    val_loss = d_val['loss/val_loss_']
                    print(d_val['print_str'])
                    best_model = val_loss < best_val_loss

                    if best_model:
                        print(f'Iter [{i_iter:d}] best model saved {val_loss:.6f} <= {best_val_loss:.6f}')
                        best_val_loss = val_loss
                        self.save_model(model, logdir, best=True)
                        # model_best = copy.deepcopy(model)
                    
                    d_eval = model.eval_step(test_loader, device=self.device, **self.training_cfg)
                    logger.add_val(i_iter, d_eval)
                    print_str = f'Iter [{i_iter:d}]'
                    for key, val in d_eval.items():
                        if key.endswith('_'):
                            print_str = print_str + f'\t{key[:-1]}: {val:.4f}'
                    print(print_str)
                    
                if i_iter % cfg.visualize_interval == 0:
                    d_val = model.visualization_step(device=self.device, env=kwargs['env'], train_dl=train_loader, test_dl=test_loader, **self.training_cfg)
                    logger.add_val(i_iter, d_val)
        
        self.save_model(model, logdir, i_iter="last")
        
        return model, best_val_loss, i_iter

    def save_model(self, model, logdir, best=False, i_iter=None, i_epoch=None):
        if best:
            pkl_name = "model_best.pkl"
        else:
            if i_iter is not None:
                pkl_name = f"model_iter_{i_iter}.pkl"
            else:
                pkl_name = f"model_epoch_{i_epoch}.pkl"
        state = {"epoch": i_epoch, "iter": i_iter, "model_state": model.state_dict()}
        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)
        print(f"Model saved: {pkl_name}")

class PairwiseNetTrainer:
    def __init__(self, optimizer, cfg, device, **kwargs):
        self.cfg = cfg
        self.training_cfg = self.cfg.training
        self.device = device
        self.optimizer = optimizer
        self.criterion = get_loss(self.training_cfg['loss'])
        
    def train(self, model, d_dataloaders, logger=None, logdir="", **kwargs):
        cfg = self.training_cfg
    
        time_meter = averageMeter()
        train_loader, val_loader, test_loader = (d_dataloaders["training"], d_dataloaders["validation"], d_dataloaders["test"])
        i_iter = kwargs.get('iter_bias', 0)
        best_val_loss = np.inf
    
        for i_epoch in range(1, cfg['n_epoch'] + 1):
            for pcd1, pcd2, SE3, y in train_loader:
                i_iter += 1

                model.train()
                pcd1, pcd2, SE3, y = pcd1.to(self.device), pcd2.to(self.device), SE3.to(self.device), y.to(self.device)
                start_ts = time.time()
                d_train = model.train_step(pcd1, pcd2, SE3, y, self.criterion, optimizer=self.optimizer, **kwargs)
                time_meter.update(time.time() - start_ts)
                logger.process_iter_train(d_train)

                if i_iter % cfg.print_interval == 0:
                    d_train = logger.summary_train(i_iter, reset=True)
                    print(
                        f"{now()} Epoch [{i_epoch:d}] Iter [{i_iter:d}]\tAvg Loss: {d_train['loss/train_loss_']:.6f}\tElapsed time: {time_meter.sum:.4f}\t"
                    )
                    time_meter.reset()
                    
                model.eval()
                if i_iter % cfg.val_interval == 0:
                    for pcd1, pcd2, SE3, y in val_loader:
                        pcd1, pcd2, SE3, y = pcd1.to(self.device), pcd2.to(self.device), SE3.to(self.device), y.to(self.device)
                        d_val = model.validation_step(pcd1, pcd2, SE3, y, self.criterion)
                        logger.process_iter_val(d_val)
                    d_val = logger.summary_val(i_iter, reset=True)
                    val_loss = d_val['loss/val_loss_']
                    print(d_val['print_str'])
                    best_model = val_loss < best_val_loss

                    if best_model:
                        print(f'Iter [{i_iter:d}] best model saved {val_loss:.6f} <= {best_val_loss:.6f}')
                        best_val_loss = val_loss
                        self.save_model(model, logdir, best=True)
                    
                    d_eval = model.eval_step(test_dl=test_loader, env=kwargs['env'], cfg=self.cfg, device=self.device, **self.training_cfg)
                    logger.add_val(i_iter, d_eval)
                    print_str = f'Iter [{i_iter:d}]'
                    for key, val in d_eval.items():
                        if key.endswith('_'):
                            print_str = print_str + f'\t{key[:-1]}: {val:.4f}'
                    print(print_str)
                    
                if i_iter % cfg.visualize_interval == 0:
                    d_val = model.visualization_step(test_dl=test_loader, env=kwargs['env'], cfg=self.cfg, device=self.device, **self.training_cfg)
                    logger.add_val(i_iter, d_val)
        
        self.save_model(model, logdir, i_iter="last")
        
        return model, best_val_loss, i_iter

    def save_model(self, model, logdir, best=False, i_iter=None, i_epoch=None):
        if best:
            pkl_name = "model_best.pkl"
        else:
            if i_iter is not None:
                pkl_name = f"model_iter_{i_iter}.pkl"
            else:
                pkl_name = f"model_epoch_{i_epoch}.pkl"
        state = {"epoch": i_epoch, "iter": i_iter, "model_state": model.state_dict()}
        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)
        print(f"Model saved: {pkl_name}")