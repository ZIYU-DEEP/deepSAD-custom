from pathlib import Path
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision.datasets.utils import download_url

import os
import torch
import numpy as np


class CustomDataset(Dataset):
    """
    ODDSDataset class for datasets from Outlier Detection DataSets (ODDS): http://odds.cs.stonybrook.edu/

    Dataset class with additional targets for the semi-supervised setting and modification of __getitem__ method
    to also return the semi-supervised target as well as the index of a data sample.
    """
    def __init__(self, root: str, dataset_name: str, train=True,
                 random_state=None, simulate=False, normal_data_file="",
                 abnormal_data_file=""):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.train = train  # training set or test set
        self.normal_file_name = normal_data_file
        self.normal_data_file = self.root / self.normal_file_name

        if abnormal_data_file:
            self.abnormal_file_name = abnormal_data_file
            self.abnormal_data_file = self.root / self.abnormal_file_name

        normal_data = np.load(self.normal_data_file)
        print('Normal data: I am loaded!')
        y_normal = np.zeros(normal_data.shape[0])
        X_train_norm, X_test_norm, y_train_norm, y_test_norm = \
            train_test_split(normal_data, y_normal, test_size=0.2,
                             random_state=random_state)

        if abnormal_data_file:
            abnormal_data = np.load(self.abnormal_data_file)
            print('Abnormal data: I am loaded!')
            y_abnormal = np.ones(abnormal_data.shape[0])
            X_train_out, X_test_out, y_train_out, y_test_out = \
                train_test_split(abnormal_data, y_abnormal, test_size=0.2,
                                 random_state=random_state)
            X_train = np.concatenate((X_train_norm, X_train_out))
            X_test = np.concatenate((X_test_norm, X_test_out))
            y_train = np.concatenate((y_train_norm, y_train_out))
            y_test = np.concatenate((y_test_norm, y_test_out))
        else:
            X_train, X_test, y_train, y_test = X_train_norm, X_test_norm, \
                                               y_train_norm, y_test_norm

        if self.train:
            self.data = torch.tensor(X_train, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
        else:
            self.data = torch.tensor(X_test, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.int64)

        self.semi_targets = self.targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
        sample, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])
        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)
