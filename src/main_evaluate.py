import os
import sys
import numpy as np
import pandas as pd
from DeepSAD_eval import DeepSAD_eval
from datasets.main_eval import load_dataset_eval


train_source = str(sys.argv[1])  # e.g. JCL
eval_source = str(sys.argv[2])  # e.g. downtown_319
fp_rate = float(sys.argv[3])  # e.g. 0.05

net_name = 'custom_lstm'
result_path = '/net/adv_spectrum/SADlog_eval/train_{}_eval_{}'.format(
    train_source, eval_source)
data_path = '/net/adv_spectrum/array_data'
normal_class = 0
num_threads = 0
n_jobs_dataloader = 0
load_config = True
device = 'cuda:1'
normal_data_file = 'ryerson_train_normal.npy'
abnormal_data_file = 'ryerson_ab_train_{}_abnormal.npy'.format(eval_source)
random_state = 42
load_model = '/net/adv_spectrum/SADlog/{}/model.tar'.format(train_source)
txt_result_path = '/net/adv_spectrum/SADlog_eval/result_txt'
txt_result_file = txt_result_path + '/full_results_eval.txt'

if not os.path.exists(result_path):
    print('Detect new result path!')
    os.makedirs(result_path)
    print('Set up the result path!')

if not os.path.exists(txt_result_path):
    print('Detect new txt result path!')
    os.makedirs(txt_result_path)
    print('Set up the txt result path!')

dataset = load_dataset_eval(data_path, normal_data_file,
                            abnormal_data_file, random_state)

deepSAD_eval = DeepSAD_eval(1)
deepSAD_eval.set_network(net_name)
deepSAD_eval.load_model(model_path=load_model, map_location=device)

# Test model
print('start evaluation!')
deepSAD_eval.evaluate(dataset, device=device,
                      n_jobs_dataloader=n_jobs_dataloader)
print('evaluation finished!')

# Save results, model, and configuration
deepSAD_eval.save_results(export_json=result_path + '/results.json')
print('results saved!')

# Generate result DataFrame
print('get the dataframe!')
train_auc = deepSAD_eval.results['test_auc']
indices, labels, scores = zip(*deepSAD_eval.results['test_scores'])
indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
result_df = pd.DataFrame()
result_df['indices'] = indices
result_df['labels'] = labels
result_df['scores'] = scores
result_df_path = '{}/result_df_train_{}_eval_{}.pkl'.format(result_path,
                                                            train_source,
                                                            eval_source)
result_df.to_pickle(result_df_path)

# Write the file for detection rate
result_df.drop('indices', inplace=True, axis=1)
df_normal = result_df[result_df.labels == 0]
df_abnormal = result_df[result_df.labels == 1]
cut = df_normal.scores.quantile(1 - fp_rate)
y = [1 if e > cut else 0 for e in df_abnormal['scores'].values]
f = open(txt_result_file, 'a')
f.write('=====================\n')
f.write('[DataFrame Name] {}\n'.format(result_df_path))
f.write('[Normal to Abnormal Ratio] 1:{}\n'
        .format(len(df_abnormal) / len(df_normal)))
f.write('[Train AUC] {}\n'.format(train_auc))
f.write('[Detection Rate] {}\n'.format(sum(y) / len(y)))
f.write('[FP Rate] {}\n'.format(fp_rate))
f.write('=====================\n\n')
f.close()
print('[Detection Rate] {}\n'.format(sum(y) / len(y)))
print('finished!')
