# -*- coding: utf-8 -*-
"""
@简介：根据已有的特征，使用多项式方法构造出更多特征
@author: Jian
"""
import pickle
import time
from sklearn.preprocessing import PolynomialFeatures

t_start = time.time()

"""读取原特征"""
features_path = './data_s_lsvc_l2_143w_lsa.pkl'
f = open(features_path, 'rb')
x_train, y_train, x_test = pickle.load(f)
f.close()

"""使用多项式方法构造出更多的特征"""
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)#degree控制多项式最高次数
x_train_new = poly.fit_transform(x_train)
x_test_new = poly.transform(x_test)

"""将构造好的特征保存至本地"""
data = (x_train_new, y_train,  x_test_new)
features_constr_path = features_path.split('/')[-1] + '_constr.pkl'
f_data = open(features_constr_path, 'wb')
pickle.dump(data, f_data)
f_data.close()

t_end = time.time()
print("构造特征完成，共耗时：{}min".format((t_end-t_start)/60))



