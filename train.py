import numpy as np
import torch
import os

import argparse
from omegaconf import OmegaConf
import warnings
warnings.simplefilter("ignore")

from tensorboardX import SummaryWriter
from datetime import datetime

from training.model import get_model
from training.optimizers import get_optimizer
from training.trainers import get_trainer, get_logger
from training.loader import get_dataloader
from envs import get_env
from utils import save_yaml

torch.multiprocessing.set_sharing_strategy('file_system')

def parse_arg_type(val):
    if val.isnumeric():
        return int(val)
    if (val == 'True') or (val == 'true'):
        return True
    if (val == 'False') or (val == 'false'):
        return False
    try:
        return float(val)
    except:
        return str(val)

def parse_unknown_args(l_args):
    """convert the list of unknown args into dict
    this does similar stuff to OmegaConf.from_cli()
    I may have invented the wheel again..."""
    n_args = len(l_args) // 2
    kwargs = {}
    for i_args in range(n_args):
        key = l_args[i_args*2]
        val = l_args[i_args*2 + 1]
        assert '=' not in key, 'optional arguments should be separated by space'
        kwargs[key.strip('-')] = parse_arg_type(val)
    return kwargs

def parse_nested_args(d_cmd_cfg):
    """produce a nested dictionary by parsing dot-separated keys
    e.g. {key1.key2 : 1}  --> {key1: {key2: 1}}"""
    d_new_cfg = {}
    for key, val in d_cmd_cfg.items():
        l_key = key.split('.')
        d = d_new_cfg
        for i_key, each_key in enumerate(l_key):
            if i_key == len(l_key) - 1:
                d[each_key] = val
            else:
                if each_key not in d:
                    d[each_key] = {}
                d = d[each_key]
    return d_new_cfg

def run(cfg, writer):
    seed = cfg.training.get('seed', 1)
    print(f"running with random seed : {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
    device = cfg.device
    
    env = get_env(cfg.env, device=device)
    
    model = get_model(cfg.model, env=env).to(device)
    logger = get_logger(cfg, writer)
    
    d_dataloaders = {}
    for key, dataloader_cfg in cfg.data.items():
        d_dataloaders[key] = get_dataloader(dataloader_cfg)
    
    optimizer = get_optimizer(cfg.training.optimizer, model.parameters())
    trainer = get_trainer(optimizer, cfg)

    model, best_val_loss, i_iter = trainer.train(
        model,
        d_dataloaders,
        logger=logger,
        logdir=writer.file_writer.get_logdir(),
        env=env,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--device", default=1)
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--run", default=None, type=str)
    args, unknown = parser.parse_known_args()
    d_cmd_cfg = parse_unknown_args(unknown)
    d_cmd_cfg = parse_nested_args(d_cmd_cfg)
    print(d_cmd_cfg)
    
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(cfg, d_cmd_cfg)
    env_cfg = OmegaConf.load(os.path.join(cfg.data.test.root, 'env_config.yml'))
    cfg['env'] = env_cfg
    cfg.logger.project = cfg.env.id

    if args.device == "cpu":
        cfg["device"] = f"cpu"
    else:
        cfg["device"] = f"cuda:{args.device}"

    if args.run is None:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    else:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S_") + args.run
    cfg.logger.run_id = run_id
    
    config_basename = os.path.basename(args.config).split(".")[0]
    
    if hasattr(cfg, "logdir"):
        logdir = cfg["logdir"]
    else:
        logdir = args.logdir
    logdir = os.path.join(logdir, run_id)
    writer = SummaryWriter(logdir=logdir)
    print("Result directory: {}".format(logdir))

    # copy config file
    copied_yml = os.path.join(logdir, os.path.basename(args.config))
    cfg.logger.config_file = copied_yml
    save_yaml(copied_yml, OmegaConf.to_yaml(cfg))
    
    print(OmegaConf.to_yaml(cfg))
    print(f"config saved as {copied_yml}")

    run(cfg, writer)