# -*- coding: utf-8 -*-
"""
@brief : 将原始数据数字化为hash特征，并将结果保存至本地
@author: Jian
"""
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
import pickle
import time

t_start = time.time()

"""=====================================================================================================================
1 加载原始数据
"""
df_train = pd.read_csv('../data/train_set.csv')
df_train.drop(columns='article', inplace=True)
df_test = pd.read_csv('../data/test_set.csv')
df_test.drop(columns='article', inplace=True)
df_all = pd.concat(objs=[df_train, df_test], axis=0, sort=True)
y_train = (df_train['class'] - 1).values

"""=====================================================================================================================
2 数字化为hash特征
"""
vectorizer = HashingVectorizer(ngram_range=(1, 2), n_features=200)
d_all = vectorizer.fit_transform(df_all['word_seg'])
x_train = d_all[:len(y_train)]
x_test = d_all[len(y_train):]

"""=====================================================================================================================
3 将hsah特征保存至本地
"""
data = (x_train.toarray(), y_train, x_test.toarray())
f_data = open('./data_hash.pkl', 'wb')
pickle.dump(data, f_data)
f_data.close()

t_end = time.time()
print("已将原始数据数字化为hash特征，共耗时：{}min".format((t_end-t_start)/60))


