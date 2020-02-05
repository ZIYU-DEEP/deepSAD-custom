python create_squeeze_array.py abnormal ryerson_ab_train_LOS-5M-USRP2
python create_squeeze_array.py abnormal ryerson_ab_train_LOS-5M-USRP3
python create_squeeze_array.py abnormal ryerson_ab_train_NLOS-5M-USRP1
python create_squeeze_array.py abnormal ryerson_ab_train_Dynamics-5M-USRP1

python main_evaluate.py sigOver_319 LOS-5M-USRP2_1599 0.05
python main_evaluate.py sigOver_319 LOS-5M-USRP3_1599 0.05
python main_evaluate.py sigOver_319 NLOS-5M-USRP1_1599 0.05
python main_evaluate.py sigOver_319 Dynamics-5M-USRP1_1599 0.05
