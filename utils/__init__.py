from tqdm.auto import tqdm
import time, yaml

class progress_tracker():
    def __init__(self, **kwargs):
        self.tqdm_obj = tqdm(**kwargs)
        self.value_old = 0
        
    def update(self, x):
        self.tqdm_obj.update(x - self.value_old)
        self.value_old = x
        self.tqdm_obj.refresh()
    
    def close(self):
        self.tqdm_obj.close()

class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def now():
    return f'[{time.strftime("%Y/%m/%d %X", time.localtime())}]'

def save_yaml(filename, text):
    """parse string as yaml then dump as a file"""
    with open(filename, "w") as f:
        yaml.dump(yaml.safe_load(text), f, default_flow_style=False)