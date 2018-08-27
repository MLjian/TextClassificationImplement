# -*- coding: utf-8 -*-
"""
@brief : 将官方的训练集划分出一部分作为验证集，并训练结束后查看验证集上的分数，方便手动调参
@author: Jian
"""
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
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
 
"""划分训练集和验证集，验证集比例为test_size"""
x_train, x_vali, y_train, y_vali = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

"""训练分类器"""
classifier = LinearSVC()
classifier.fit(x_train, y_train)

"""对验证集进行预测，并计算验证集的得分"""
pre_vali = classifier.predict(x_vali)
score_vali = f1_score(y_vali, pre_vali, average='macro')
print("验证集分数：{}".format(score_vali))

"""根据上面训练好的分类器对测试集的每个样本进行预测"""
y_test = classifier.predict(x_test) 

"""将测试集的预测结果保存至本地"""
df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:, ['id', 'class']]
df_result.to_csv('../results/beginner.csv', index=False)

t_end = time.time()
print("训练结束，耗时:{}s".format(t_end-t_start))
