# -*- coding: utf-8 -*-
"""
@brief : 将tfidf特征降维为lsa特征，并将结果保存至本地
@author: Jian
"""
from sklearn.decomposition import TruncatedSVD
import pickle
import time

t_start = time.time()

"""读取tfidf特征"""
tfidf_path = './data_tfidf_selected_lsvc_l2_143w.pkl'
f_tfidf = open(tfidf_path, 'rb')
x_train, y_train, x_test = pickle.load(f_tfidf)
f_tfidf.close()

"""特征降维：lsa"""
print("lsa......")
lsa = TruncatedSVD(n_components=200)
x_train = lsa.fit_transform(x_train)
x_test = lsa.transform(x_test)

"""将lsa特征保存至本地"""
data = (x_train, y_train, x_test)
f_data = open('./data_s_lsvc_l2_143w_lsa.pkl', 'wb')
pickle.dump(data, f_data)
f_data.close()

t_end = time.time()
print("lsa特征完成，共耗时：{}min".format((t_end-t_start)/60))
