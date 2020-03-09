python main.py custom custom_lstm /net/adv_spectrum/SADlog/downtown_big_sigOver_mini_75 /net/adv_spectrum/array_data --eta 0.75 --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 60 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class 0 --known_outlier_class 1 --n_known_outlier_classes 1 --seed 42 --normal_data_file downtown_big_normal.npy --abnormal_data_file downtown_sigOver_10ms_mini_abnormal.npy --txt_result_file /net/adv_spectrum/SADlog/result_txt/1_full_results.txt

python main_evaluate.py downtown_big_sigOver_mini_75 downtown_sigOver_10ms_big downtown_big_normal.npy downtown_sigOver_10ms_big_abnormal.npy 0.05

python main_evaluate.py downtown_big_sigOver_mini_75 downtown_LOS-5M-USRP1_half downtown_big_normal.npy downtown_LOS-5M-USRP1_half_abnormal.npy 0.05

python main_evaluate.py downtown_big_sigOver_mini_75 downtown_LOS-5M-USRP2_half downtown_big_normal.npy downtown_LOS-5M-USRP2_half_abnormal.npy 0.05

python main_evaluate.py downtown_big_sigOver_mini_75 downtown_LOS-5M-USRP3_half downtown_big_normal.npy downtown_LOS-5M-USRP3_half_abnormal.npy 0.05

python main_evaluate.py downtown_big_sigOver_mini_75 downtown_NLOS-5M-USRP1_half downtown_big_normal.npy downtown_NLOS-5M-USRP1_half_abnormal.npy 0.05

python main_evaluate.py downtown_big_sigOver_mini_75 downtown_Dynamics-5M-USRP1_half downtown_big_normal.npy downtown_Dynamics-5M-USRP1_half_abnormal.npy 0.05
