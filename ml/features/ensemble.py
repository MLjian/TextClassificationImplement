# -*- coding: utf-8 -*-
"""
@brief : tfidf/doc2vec两种特征进行特征融合，并将结果保存至本地
@author: Jian
"""
import pandas as pd
import numpy as np
import pickle
import time

t_start = time.time()

f1 = open('./data_tfidf_1000.pkl', 'rb')
x_train_1, y_train, x_test_1 = pickle.load(f1)
f1.close()

f2 = open('./data_lsa.pkl', 'rb')
x_train_2, _, x_test_2 = pickle.load(f2)
f2.close()

f3 = open('./data_doc2vec.pkl', 'rb')
x_train_3, _, x_test_3 = pickle.load(f3)
f3.close()


x_train = np.concatenate((x_train_1.toarray(), x_train_2, x_train_3), axis=1)
x_test = np.concatenate((x_test_1.toarray(), x_test_2, x_test_3), axis=1)

"""=====================================================================================================================
3 保存至本地
"""
data = (x_train, y_train, x_test)
fp = open('./data_ensemble.pkl', 'wb')
pickle.dump(data, fp)
fp.close()

t_end = time.time()
print("已将原始数据数字化为融合的特征，共耗时：{}min".format((t_end-t_start)/60))
