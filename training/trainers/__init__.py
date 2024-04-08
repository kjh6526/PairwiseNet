from training.trainers.trainer import BaseTrainer, PairwiseNetTrainer
from training.trainers.logger import BaseLogger

def get_trainer(optimizer, cfg):
    trainer_type = cfg.get("trainer", "base")
    device = cfg["device"]
    if trainer_type == "base":
        trainer = BaseTrainer(optimizer, cfg.training, device=device)
    elif trainer_type == 'PairwiseNet':
        trainer = PairwiseNetTrainer(optimizer, cfg, device=device)
    return trainer

def get_logger(cfg, writer):
    logger_type = cfg["logger"].get("type", "base")
    endwith = cfg["logger"].get("endwith", [])
    wandblog = cfg["logger"].get("wandblog", False)
    if logger_type in ["base"]:
        logger = BaseLogger(writer, cfg=cfg, endwith=endwith, wandblog=wandblog)
    return logger