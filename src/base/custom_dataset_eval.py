from pathlib import Path
from torch.utils.data import Dataset

import os
import torch
import numpy as np


class CustomDataset_eval(Dataset):
    def __init__(self,
                 root: str,
                 random_state=None,
                 normal_data_file='skip',
                 abnormal_data_file=''):

        super(Dataset, self).__init__()

        self.classes = [0, 1]
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)

        self.root = Path(root)

        if normal_data_file != 'skip':
            self.normal_file_name = normal_data_file
            self.abnormal_file_name = abnormal_data_file
            self.normal_data_file = self.root / self.normal_file_name
            self.abnormal_data_file = self.root / self.abnormal_file_name
            X_normal = np.load(self.normal_data_file)
            print('Normal data: I am loaded!')
            y_normal = np.zeros(X_normal.shape[0])

        X_abnormal = np.load(self.abnormal_data_file)
        print('Abnormal data: I am loaded!')
        y_abnormal = np.ones(X_abnormal.shape[0])

        if normal_data_file != 'skip':
            X = np.concatenate((X_normal, X_abnormal))
            y = np.concatenate((y_normal, y_abnormal))
        else:
            X = X_abnormal
            y = y_abnormal

        self.data = torch.tensor(X, dtype=torch.float32)
        self.targets = torch.tensor(y, dtype=torch.int64)
        self.semi_targets = self.targets

    def __getitem__(self, index):
        sample, target, semi_target = self.data[index], int(
            self.targets[index]), int(self.semi_targets[index])
        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)
