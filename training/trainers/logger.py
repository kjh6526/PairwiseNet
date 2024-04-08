#############################################
#                                           #
# code from IRVAE_public (Lee et al., 2022) #
#                                           #
#############################################

from utils import averageMeter
import wandb

class BaseLogger:
    """BaseLogger that can handle most of the logging
    logging convention
    ------------------
    'loss' has to be exist in all training settings
    endswith('_') : scalar
    endswith('@') : image
    """
    def __init__(self, tb_writer, cfg, endwith=[], wandblog=False):
        """tb_writer: tensorboard SummaryWriter"""
        self.writer = tb_writer
        self.endwith = endwith
        self.train_loss_meter = averageMeter()
        self.val_loss_meter = averageMeter()
        self.d_train = {}
        self.d_val = {}
        self.wandblog = wandblog
        self.cfg = cfg
        if self.wandblog:
            config = {
                'epochs': self.cfg.training.n_epoch,
                'seed': self.cfg.training.seed,
                'train_batch' : self.cfg.data.training.batch_size,
                'lr' : self.cfg.training.optimizer.lr
                #ADD Confing to Sweep for auto parameter tuning
            }
            wandb.init(project=self.cfg.id, config=config)
            wandb.save("configs/training/*")

    def process_iter_train(self, d_result):
        self.train_loss_meter.update(d_result['loss'])
        self.d_train = d_result

    def summary_train(self, i, reset=True):
        self.d_train['loss/train_loss_'] = self.train_loss_meter.avg 
        for key, val in self.d_train.items():
            if key.endswith('_'):
                self.writer.add_scalar(key, val, i)
            if key.endswith('@') and ('@' in self.endwith):
                if val is not None:
                    self.writer.add_image(key, val, i)

        result = self.d_train
        self.d_train = {}
        if reset:
            self.reset_train()
        if self.wandblog:
            wandb.log(result, step=i) #log summary loss 
        return result

    def process_iter_val(self, d_result):
        self.val_loss_meter.update(d_result['loss'])
        self.d_val = d_result

    def summary_val(self, i, d_val=None, reset=True):
        if d_val is None:
            d_val = self.d_val
            d_val['loss/val_loss_'] = self.val_loss_meter.avg 
        l_print_str = [f'Iter [{i:d}]']
        for key, val in d_val.items():
            if key.endswith('_'):
                self.writer.add_scalar(key, val, i)
                l_print_str.append(f'\t{key[:-1]}: {val:.4f}')
            if key.endswith('@') and ('@' in self.endwith):
                if val is not None:
                    self.writer.add_image(key, val, i)
        if self.wandblog: 
            wandb.log(d_val, step=i) #log evaluation loss
        print_str = ' '.join(l_print_str)

        result = d_val
        result['print_str'] = print_str
        self.d_val = {}
        
        if reset:
            self.reset_val()
        
        return result

    def reset_val(self):
        self.val_loss_meter.reset()
        
    def reset_train(self):
        self.train_loss_meter.reset()

    def add_val(self, i, d_result):
        for key, val in d_result.items():
            if key.endswith('_'):
                self.writer.add_scalar(key, val, i)
            if key.endswith('@') and ('@' in self.endwith):
                if val is not None:
                    self.writer.add_image(key, val, i)
        
        if self.wandblog:
            wandb.log(d_result, step=i) #log evaluation result