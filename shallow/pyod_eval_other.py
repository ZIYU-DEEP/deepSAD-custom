import os
import sys
import pyod
import glob
import joblib
import numpy as np
import multiprocessing as mp

from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.pca import PCA
from pyod.models.iforest import IForest

filename = str(sys.argv[1])  # e.g. '0208_anomaly'

model_path = '/net/adv_spectrum/shallow/pyod/model/ryerson_train/'
data_path = '/net/adv_spectrum/array_data/'
data_file = '{}{}_abnormal.npy'.format(data_path, filename)
result_path = '/net/adv_spectrum/shallow/pyod/result/'
result_file = '{}result_{}.txt'.format(result_path, filename)

# 1. Load model
model_path_list = sorted(glob.glob(model_path + '*.pkl'))
model_dict = {'IForest': {'path': None, 'clf': None},
              'KNN': {'path': None, 'clf': None},
              'LOF': {'path': None, 'clf': None},
              'OCVSM': {'path': None, 'clf': None},
              'PCA': {'path': None, 'clf': None}}

for i, key in enumerate(model_dict):
    print(i, key)
    model_dict[key]['path'] = model_path_list[i]
    model_dict[key]['clf'] = joblib.load(model_path_list[i])

print('All models have been loaded!')


# 2. Load data_file
data = np.load(data_file)
data = data.reshape(data.shape[0], -1)
print('Data has been loaded!')


# 3. Evaluate models
f = open(result_file, 'a')
f.write('Data Source: {}\n\n'.format(data_file))

for key in model_dict:
    clf = model_dict[key]['clf']
    y_pred = clf.predict(data)
    detection_result = y_pred.sum() / len(y_pred)
    info = 'Model Name: {}; Detection Result: {}\n'.format(key, str(detection_result))
    print(info)
    f.write(info)

f.close()
print('Done!')
