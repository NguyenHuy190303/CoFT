import os
import torch
import numpy as np
from torch.utils.data import Dataset
from dataloader.augmentations import DataTransform_TD

# Import safe_torch_load from main module
import sys
import importlib.util

def safe_torch_load(filepath, device=None, **kwargs):
    """Safe torch.load wrapper that avoids weights_only for compatibility"""
    if device:
        return torch.load(filepath, map_location=device, **kwargs)
    else:
        return torch.load(filepath, **kwargs)

class Load_Dataset(Dataset):
    def __init__(self, dataset, config, training_mode, enable_coft=False):
        self.training_mode = training_mode
        self.enable_coft = enable_coft
        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        # Keep data on CPU for DataLoader
        self.x_data = self.x_data.cpu()
        self.y_data = self.y_data.cpu()

        self.len = X_train.shape[0]
        
        # Pre-compute augmentations for self-supervised modes
        if training_mode == "self_supervised" or training_mode == "SupCon":
            self.aug1, self.aug2 = DataTransform_TD(self.x_data, config, enable_coft)
            # Safe conversion to tensors
            if isinstance(self.aug1, np.ndarray):
                self.aug1 = torch.from_numpy(self.aug1)
            if isinstance(self.aug2, np.ndarray):
                self.aug2 = torch.from_numpy(self.aug2)
            # Ensure CPU placement
            if hasattr(self.aug1, 'cpu'):
                self.aug1 = self.aug1.cpu()
            if hasattr(self.aug2, 'cpu'):
                self.aug2 = self.aug2.cpu()

    def __getitem__(self, index):
        if self.training_mode == "self_supervised" or self.training_mode == "SupCon":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, configs, training_mode, enable_coft=False):
    batch_size = configs.batch_size
    
    # Simple, stable DataLoader configuration
    num_workers = 2  # Conservative setting for stability
    pin_memory = False  # Avoid threading issues
    persistent_workers = False  # Avoid worker lifecycle issues

    if "_1p" in training_mode:
        train_dataset = safe_torch_load(os.path.join(data_path, "train_1p.pt"))
    elif "_5p" in training_mode:
        train_dataset = safe_torch_load(os.path.join(data_path, "train_5p.pt"))
    elif "_10p" in training_mode:
        train_dataset = safe_torch_load(os.path.join(data_path, "train_10p.pt"))
    elif "_50p" in training_mode:
        train_dataset = safe_torch_load(os.path.join(data_path, "train_50p.pt"))
    elif "_75p" in training_mode:
        train_dataset = safe_torch_load(os.path.join(data_path, "train_75p.pt"))
    elif "SupCon" in training_mode:
        train_dataset = safe_torch_load(os.path.join(data_path, "pseudo_train_data.pt"))
    else:
        train_dataset = safe_torch_load(os.path.join(data_path, "train.pt"))

    valid_dataset = safe_torch_load(os.path.join(data_path, "val.pt"))
    test_dataset = safe_torch_load(os.path.join(data_path, "test.pt"))

    train_dataset = Load_Dataset(train_dataset, configs, training_mode, enable_coft)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode, enable_coft)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode, enable_coft)

    if train_dataset.__len__() < batch_size:
        batch_size = 16

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        drop_last=configs.drop_last, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        drop_last=configs.drop_last, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        drop_last=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    
    return train_loader, valid_loader, test_loader