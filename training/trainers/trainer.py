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
        best_eval_metric = {}
    
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
                        self.save_model(model, logdir, best=True, metric='val_loss')
                    
                    d_eval = model.eval_step(test_dl=test_loader, env=kwargs['env'], cfg=self.cfg, device=self.device, **self.training_cfg)
                    logger.add_val(i_iter, d_eval)
                    print_str = f'Iter [{i_iter:d}]'
                    for key, val in d_eval.items():
                        if key.endswith('_'):
                            print_str = print_str + f'\t{key[:-1]}: {val:.4f}'
                            
                            if key not in best_eval_metric.keys():
                                best_eval_metric[key] = val
                            elif key in ['eval/accuracy_', 'eval/AUROC_']:
                                if val > best_eval_metric[key]:
                                    best_eval_metric[key] = val
                                    self.save_model(model, logdir, best=True, metric=key[5:-1])
                            elif key in ['eval/mse_', 'eval/safe_FPR_']:
                                if val < best_eval_metric[key]:
                                    best_eval_metric[key] = val
                                    self.save_model(model, logdir, best=True, metric=key[5:-1])
                    print(print_str)
                    
                if i_iter % cfg.visualize_interval == 0:
                    d_val = model.visualization_step(test_dl=test_loader, env=kwargs['env'], cfg=self.cfg, device=self.device, **self.training_cfg)
                    logger.add_val(i_iter, d_val)
        
        model_path = self.save_model(model, logdir, i_iter="last")
        logger.save_model(model_path)
        
        return model, best_val_loss, i_iter

    def save_model(self, model, logdir, best=False, i_iter=None, i_epoch=None, metric='val_loss'):
        if best:
            pkl_name = f"model_best_{metric}.pkl"
        else:
            if i_iter is not None:
                pkl_name = f"model_iter_{i_iter}.pkl"
            else:
                pkl_name = f"model_epoch_{i_epoch}.pkl"
        state = {"epoch": i_epoch, "iter": i_iter, "model_state": model.state_dict()}
        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)
        print(f"Model saved: {pkl_name}")
        return save_path


class PcdEncoderTrainer:
    """Trainer for a conventional iterative training of model"""
    def __init__(self, optimizer, training_cfg, device):
        self.cfg = training_cfg
        self.device = device
        self.optimizer = optimizer
        self.d_val_result = {}
        self.loss = get_loss(training_cfg['loss'], device=device)

    def train(self, model, d_dataloaders, logger=None, logdir='', *args, **kwargs):
        cfg = self.cfg
        opt = self.optimizer
        # show_latent = cfg.get('show_latent', True)
        best_val_loss = np.inf
        best_classify_acc = 0
        time_meter = averageMeter()
        i = 0
        train_loader = d_dataloaders['training']
        do_val = False
        if 'validation' in d_dataloaders.keys():
            val_loader = d_dataloaders['validation']
            do_val = True

        for i_epoch in range(cfg.n_epoch):
            for output in train_loader:
                
                i += 1

                model.train()
                x = output.to(self.device)
                
                start_ts = time.time()
                d_train, recon_x = model.train_step(x, optimizer=opt, criterion=self.loss)

                time_meter.update(time.time() - start_ts)
                logger.process_iter_train(d_train)

                if i % cfg.print_interval == 0:
                    d_train = logger.summary_train(i)
                    print(f"Iter [{i:d}] Avg Loss: {d_train['loss/train_loss_']:.4f} Elapsed time: {time_meter.sum:.4f}")
                    time_meter.reset()
                
                if do_val:
                    if i % cfg.val_interval == 0:
                        model.eval()
                        for output in val_loader:
                            x = output.to(self.device)
                            d_val = model.validation_step(x, criterion=self.loss)
                            logger.process_iter_val(d_val)
                        d_val = logger.summary_val(i)
                        val_loss = d_val['loss/val_loss_']
                        print(d_val['print_str'])
                        best_model = val_loss < best_val_loss

                        # if i % cfg.save_interval == 0 or best_model:
                        #     self.save_model(model, logdir, best=best_model, i_iter=i)
                        if best_model:
                            print(f'Iter [{i:d}] best model saved {val_loss} <= {best_val_loss}')
                            best_val_loss = val_loss
                            self.save_model(model, logdir, best=best_model, i_iter=i)
                
                if i % cfg.visualize_interval == 0:
                    d_viz = model.visualization_step(x[0], recon_x[0])
                    logger.add_val(i, d_viz)
                
            if 'classification_interval_epoch' in cfg.keys():
                if (i_epoch+1) % cfg.classification_interval_epoch == 0:
                    tic = time.time()
                    model.eval()
                    classify_dict = model.classification_step(train_loader, val_loader, self.device)
                    classify_acc = classify_dict['classify_acc_']
                    logger.writer.add_scalar('classify_acc_', classify_acc, i)
                    best_classify = best_classify_acc < classify_acc
                    if best_classify:
                        print(f'Epoch [{i_epoch:d}] best classify model saved {best_classify_acc} <= {classify_acc}')
                        best_classify_acc = classify_acc
                        self.save_model(model, logdir, best=False, i_iter=i, best_classify=True)
                    else:
                        print(f'Epoch [{i_epoch:d}] classification acc: {classify_acc}')
                    toc = time.time()
                    print(f'time spent for classification : {toc-tic} s')

            if 'interpolation_interval_epoch' in cfg.keys():
                if (i_epoch+1) % cfg.interpolation_interval_epoch == 0:
                    model.eval()
                    interpolation_dict = model.interpolation_step(train_loader, self.device)

                    logger.writer.add_mesh('interpolation_inter_class(', vertices=interpolation_dict[0].transpose(1,0).unsqueeze(0), global_step=i)
                    if len(interpolation_dict) == 2:
                        logger.writer.add_mesh('interpolation_intra_class(', vertices=interpolation_dict[1].transpose(1,0).unsqueeze(0), global_step=i)

            if i_epoch % cfg.save_interval == 0:
                self.save_model(model, logdir, best=False, i_epoch=i_epoch)
            
        self.save_model(model, logdir, best=False, last=True)
        return model, best_val_loss

    def save_model(self, model, logdir, best=False, last=False, i_iter=None, i_epoch=None, best_classify=False):
        if last:
            pkl_name = "model_last.pkl"
        else:
            if best:
                pkl_name = "model_best.pkl"
            elif best_classify:
                pkl_name = "model_best_classify.pkl"
            else:
                if i_iter is not None:
                    pkl_name = "model_iter_{}.pkl".format(i_iter)
                else:
                    pkl_name = "model_epoch_{}.pkl".format(i_epoch)
        state = {"epoch": i_epoch, "model_state": model.state_dict(), 'iter': i_iter}
        save_path = os.path.join(logdir, pkl_name)
        torch.save(state, save_path)
        print(f'Model saved: {pkl_name}')