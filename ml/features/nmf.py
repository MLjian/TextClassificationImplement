# -*- coding: utf-8 -*-
"""
@brief : 将tfidf特征降维为nmf特征，并将结果保存至本地
@author: Jian
"""
from sklearn.decomposition import NMF
import pickle
import time

t_start = time.time()

"""读取tfidf特征"""
tfidf_path = './word_seg_tfidf_(1, 3)-2036592-616882-192632-62375.pkl'
f_tfidf = open(tfidf_path, 'rb')
x_train, y_train, x_test = pickle.load(f_tfidf)
f_tfidf.close()
"""特征降维：nmf"""
#print("nmf......")
num_features = 200
nmf = NMF(n_components=num_features)
x_train = nmf.fit_transform(x_train)
x_test = nmf.transform(x_test)

"""将lsa特征保存至本地"""
data = (x_train, y_train, x_test)
data_path = tfidf_path[:-4] + '-nmf.pkl'
f_data = open(data_path, 'wb')
pickle.dump(data, f_data)
f_data.close()

t_end = time.time()
#print("nmf特征完成，共耗时：{}min".format((t_end-t_start)/60))


