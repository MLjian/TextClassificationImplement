# -*- coding: utf-8 -*-
"""
@brief : 将原始数据数字化为doc2vec特征，并将结果保存至本地
@author: Jian
"""
import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import time
import pickle

t_start = time.time()

"""=====================================================================================================================
0 辅助函数 
"""


def sentence2list(sentence):
    s_list = sentence.strip().split()
    return s_list


"""=====================================================================================================================
1 数据预处理
"""
df_train = pd.read_csv('../data/train_set.csv')
df_train.drop(columns='article', inplace=True)
df_test = pd.read_csv('../data/test_set.csv')
df_test.drop(columns='article', inplace=True)
df_all = pd.concat(objs=[df_train, df_test], axis=0, sort=True)
y_train = (df_train['class'] - 1).values

df_all['word_list'] = df_all['word_seg'].apply(sentence2list)
texts = df_all['word_list'].tolist()

"""=====================================================================================================================
2 doc2vec
"""
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(texts)]
model = Doc2Vec(documents, vector_size=200, window=10, min_count=3, workers=4, epochs=10)
docvecs = model.docvecs

x_train = []
for i in range(0, 102277):
    x_train.append(docvecs[i])
x_train = np.array(x_train)

x_test = []
for j in range(102277, 204554):
    x_test.append(docvecs[j])
x_test = np.array(x_test)

"""=====================================================================================================================
3 保存至本地
"""
data = (x_train, y_train, x_test)
fp = open('./data_doc2vec.pkl', 'wb')
pickle.dump(data, fp)
fp.close()

t_end = time.time()
print("已将原始数据数字化为doc2vec特征，共耗时：{}min".format((t_end-t_start)/60))
