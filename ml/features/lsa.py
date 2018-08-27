# -*- coding: utf-8 -*-
"""
@brief : 将原始数据数字化为lsa特征，并将结果保存至本地
@author: Jian
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import time

t_start = time.time()

"""=====================================================================================================================
1 数据预处理
"""
df_train = pd.read_csv('../data/train_set.csv')
df_train.drop(columns='article', inplace=True)
df_test = pd.read_csv('../data/test_set.csv')
df_test.drop(columns='article', inplace=True)
y_train = (df_train['class'] - 1).values

"""=====================================================================================================================
2 特征工程
"""
print("tfidf......")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=100000)
corpus_count = vectorizer.fit_transform(pd.concat(objs=[df_train, df_test], axis=0, sort=True)['word_seg'])

print("lsa......")
lsa = TruncatedSVD(n_components=200)
corpus_vectors = lsa.fit_transform(corpus_count)
x_train = corpus_vectors[:len(y_train)]
x_test = corpus_vectors[len(y_train):]

"""=====================================================================================================================
3 保存至本地
"""
data = (x_train, y_train, x_test)
fp = open('./data_lsa.pkl', 'wb')
pickle.dump(data, fp)
fp.close()

t_end = time.time()
print("已将原始数据数字化为lsa特征，共耗时：{}min".format((t_end-t_start)/60))
