# -*- coding: utf-8 -*-
"""
@brief : lda/lsa/doc2vec三种特征进行特征融合，并将结果保存至本地
@author: Jian
"""
import numpy as np
import pickle
import time

t_start = time.time()

"""=====================================================================================================================
2 读取lda/lsa/doc2vec特征，并对这三种特征进行拼接融合
"""
f1 = open('./data_lda.pkl', 'rb')
x_train_1, y_train, x_test_1 = pickle.load(f1)
f1.close()

f2 = open('./data_s_lsvc_l2_143w_lsa.pkl', 'rb')
x_train_2, y_train, x_test_2 = pickle.load(f2)
f2.close()

f3 = open('./data_doc2vec_25.pkl', 'rb')
x_train_3, _, x_test_3 = pickle.load(f3)
f3.close()

x_train = np.concatenate((x_train_1, x_train_2, x_train_3), axis=1)
x_test = np.concatenate((x_test_1, x_test_2, x_test_3), axis=1)

"""=====================================================================================================================
2 将融合后的特征，保存至本地
"""
data = (x_train, y_train, x_test)
fp = open('./data_ensemble.pkl', 'wb')
pickle.dump(data, fp)
fp.close()

t_end = time.time()
print("已将原始数据数字化为融合的特征，共耗时：{}min".format((t_end-t_start)/60))
