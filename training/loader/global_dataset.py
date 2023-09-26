import os
import torch
from torch.utils import data

class Global(torch.utils.data.Dataset):
    def __init__(self, **kwargs):
        self.root = kwargs['root']
        split = kwargs['split']
        
        data = torch.load(os.path.join(self.root, 'data_q.pt'))
        label = torch.load(os.path.join(self.root, 'label.pt'))
        
        split_train_val_test = (5/7, 1/7, 1/7)
        num_train_data = int(len(data) * split_train_val_test[0])
        num_valid_data = int(len(data) * split_train_val_test[1]) 

        if split == "training":
            data = data[:num_train_data]
            label = label[:num_train_data]
        elif split == "validation":
            data = data[num_train_data:num_train_data + num_valid_data]
            label = label[num_train_data:num_train_data + num_valid_data]
        elif split == "test":
            data = data[num_train_data + num_valid_data:]
            label = label[num_train_data + num_valid_data:]
        elif split == "all":
            pass

        self.data = data
        self.label = label
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.label[idx]
        return x, y