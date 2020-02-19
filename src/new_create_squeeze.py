import numpy as np
import sys
import glob

window_size = 1000
normality = str(sys.argv[1])  # e.g. abnormal
source = str(sys.argv[2])  # e.g. ryerson_ab_train_sigOver
abnormal_folder = str(sys.argv[3])  # e.g. ryerson_train, or 871, or downtown, or campus_drive
outname = '{}_{}.npy'.format(source, normality)
input_path = '/net/adv_spectrum/data/feature/downsample_10/{}/{}/1000_250/'.format(normality, source)
output_path = '/net/adv_spectrum/_array_data/{}/{}'.format(abnormal_folder, outname)


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

normal_len = 3199
len_1 = normal_len
len_2 = normal_len // 2
len_3 = normal_len // 10

file_name = source + '_' + normality  # e.g. ryerson_ab_train_downtown_abnormal
abnormal_path = '/net/adv_spectrum/array_data/'
input_path = abnormal_path + '{}.npy'.format(file_name)

output_path_1 = input_path.replace('_abnormal', '_3199_abnormal')
output_path_2 = output_path_1.replace('3199', '1599')
output_path_3 = output_path_2.replace('1599', '319')
print('Start processing!')


np.random.shuffle(X_full)
print('Data shuffled!')

data_1 = X_full[:len_1]
data_2 = X_full[:len_2]
data_3 = X_full[:len_3]

np.save(output_path_1, data_1)
print('Data saved at {}!'.format(output_path_1))

np.save(output_path_2, data_2)
print('Data saved at {}'.format(output_path_2))

np.save(output_path_3, data_3)
print('Data saved at {}'.format(output_path_3))
print('Done!')
