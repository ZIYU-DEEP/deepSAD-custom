import numpy as np
import os
import pickle
import joblib
import sys
import glob
import torch

window_size = 1000
normality = str(sys.argv[1])  # e.g. abnormal
source = str(sys.argv[2])  # e.g. ryerson_ab_train_sigOver
outname = '{}_{}.npy'.format(source, normality)
input_path = '/net/adv_spectrum/data/feature/downsample_10/{}/{}/1000_250/'.format(normality, source)
output_path = '/net/adv_spectrum/array_data/{}'.format(outname)


def array_to_window(X, window_size):
    """
    Inputs:
        X (np.array): Its shape should be (n_time_steps, 128)
        window_size (int): the number of time steps in a window

    Return:
        result (np.array): Its shape should be (n_windows, 1, window_size, 128)
    """
    result = []
    ind = np.arange(0, X.shape[0], window_size)

    for start, end in zip(ind, np.r_[ind[1:], X.shape[0]]):
        if end - start < window_size:
            # Discard the last few lines
            break
        result.append(X[start:end, :])

    return np.array(result)


def txt_to_series(file_path, n_channels=128):
    features = []

    with open(file_path, 'r') as f:
        for line in f:
            x = line.split()
            features.append(x)

    series = np.array(features).reshape((-1, n_channels)).astype('float64')
    return series


series_list = []
print('Start constructing normal series....')
for filename in sorted(glob.glob(input_path + '*.txt')):
    print(filename)
    series = txt_to_series(filename)
    print(series.shape)
    series_list.append(series)

X_full= array_to_window(series_list.pop(0), window_size)
for i, X in enumerate(series_list):
    print('Converting the {}th array to window...'.format(i))
    X_windowed = array_to_window(X, window_size)
    print('Concatenating...\n')
    X_full = np.concatenate((X_full, X_windowed), axis=0)

print('Done converting and concatenating!')

np.save(output_path, X_full)
print('The array has been saved at {}'.format(output_path))
