# -*- coding: utf-8 -*-

"""
@简介：对特征进行嵌入式选择
@author: Jian
"""
import time
import pickle
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC

t_start = time.time()

"""读取特征"""
features_path = './data_tf.pkl'#tfidf特征的路径
fp = open(features_path, 'rb')
x_train, y_train, x_test = pickle.load(fp)
fp.close()

"""进行特征选择"""
alo_name = 'LSVC_l2'
lsvc = LinearSVC(penalty='l2', C=1.0, dual=True).fit(x_train, y_train)
slt = SelectFromModel(lsvc, prefit=True)
x_train_s = slt.transform(x_train)
x_test_s = slt.transform(x_test)

"""保存选择后的特征至本地"""
num_features = x_train_s.shape[1]
data_path = features_path.split('.')[-2] + '_select_' + alo_name + '_' + str(num_features) + '.pkl'
data_f = open(data_path, 'wb') 
pickle.dump((x_train_s, y_train, x_test_s), data_f)
data_f.close()

t_end = time.time()
print("特征选择完成，选择{}个特征，共耗时{}min".format(num_features, (t_end-t_start)/60))



