import numpy as np
import sys

normal_len = 3199
len_1 = normal_len
len_2 = normal_len // 2
len_3 = normal_len // 10

file_name = str(sys.argv[1])  # e.g. ryerson_ab_train_downtown_abnormal
abnormal_path = '/net/adv_spectrum/array_data/'
input_path = abnormal_path + '{}.npy'.format(file_name)

output_path_1 = input_path.replace('_abnormal', '_3199_abnormal')
output_path_2 = output_path_1.replace('3199', '1599')
output_path_3 = output_path_2.replace('1599', '319')
print('Start processing!')

data = np.load(input_path)
print('Data loaded!')

np.random.shuffle(data)
print('Data shuffled!')

data_1 = data[:len_1]
data_2 = data[:len_2]
data_3 = data[:len_3]

np.save(output_path_1, data_1)
print('Data saved at {}!'.format(output_path_1))

np.save(output_path_2, data_2)
print('Data saved at {}'.format(output_path_2))

np.save(output_path_3, data_3)
print('Data saved at {}'.format(output_path_3))
print('Done!')
