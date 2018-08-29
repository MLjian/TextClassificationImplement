# -*- coding: utf-8 -*-
"""
@brief : 根据features_path中的数据，对机器学习模型进行训练，并对测试集进行预测，并将结果保存至本地
@How to use：使用前，先对sklearn_config文件进行参数配置，然后才能运行此文件进行学习训练
@author: Jian
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import time
import pickle
from sklearn_config import features_path, clf_name, clf, status_vali

t_start = time.time()

"""=====================================================================================================================
1 读取数据
"""
data_fp = open(features_path, 'rb')
x_train, y_train, x_test = pickle.load(data_fp)
data_fp.close()

"""划分训练集和验证集，验证集比例为test_size"""
if status_vali:
    x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

"""=====================================================================================================================
2 训练分类器
"""
clf.fit(x_train, y_train)

"""=====================================================================================================================
3 在验证集上评估模型
"""
if status_vali:
    pre_vali = clf.predict(x_vali)
    score_vali = f1_score(y_true=y_vali, y_pred=pre_vali, average='macro')
    print("验证集分数：{}".format(score_vali))

"""=====================================================================================================================
4 对测试集进行预测;将预测结果转换为官方标准格式；并将结果保存至本地
"""
y_test = clf.predict(x_test) + 1
df_result = pd.DataFrame(data={'id':range(102277), 'class': y_test.tolist()})
result_path = '../results/' + features_path.split('/')[-1] + '_sklearn_' + clf_name + '.csv'
df_result.to_csv(result_path, index=False)

t_end = time.time()
print("训练结束，耗时:{}min".format((t_end - t_start) / 60))
