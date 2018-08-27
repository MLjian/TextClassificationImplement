# -*- coding: utf-8 -*-
"""
@brief : 自动搜索分类器的最优的超参数值
@author: Jian
"""
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
import time

print("开始......")
t_start = time.time()

df_train = pd.read_csv('../data/train_set.csv')
df_test = pd.read_csv('../data/test_set.csv')
df_all = pd.concat(objs=[df_train, df_test], axis=0, sort=True)

vectorizer = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
vectorizer.fit(df_all['word_seg'])
x_train = vectorizer.transform(df_train['word_seg'])
x_test = vectorizer.transform(df_test['word_seg'])
y_train = (df_train['class']-1).values
print("特征工程结束！")
 

"""训练分类器"""
params = {'penalty':['l2', 'l1'], 'C':[1.0, 2.0, 3.0]}
svc = LinearSVC(dual=False)
clf = GridSearchCV(estimator=svc, param_grid=params, scoring='f1_macro', n_jobs=1, cv=3, verbose=3)
clf.fit(x_train, y_train)

"""根据上面训练好的分类器对测试集的每个样本进行预测"""
y_test = clf.predict(x_test) 

"""将测试集的预测结果保存至本地"""
df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:, ['id', 'class']]
df_result.to_csv('../results/beginner.csv', index=False)

t_end = time.time()
print("训练结束，耗时:{}s".format(t_end-t_start))


