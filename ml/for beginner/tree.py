"""
@简介：tfidf特征/ 决策树算法
@成绩： 
"""
#导入所需要的软件包
import pandas as pd
from sklearn import tree
from sklearn.feature_extraction.text import TfidfVectorizer

print("开始...............")

"""加载数据，并进行简单处理"""
df_train = pd.read_csv('../data/train_set.csv')
df_test = pd.read_csv('../data/test_set.csv')
df_train.drop(columns=['article', 'id'], inplace=True)
df_test.drop(columns=['article'], inplace=True)

"""特征工程"""
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9)
vectorizer.fit(df_train['word_seg'])
x_train = vectorizer.transform(df_train['word_seg'])
x_test = vectorizer.transform(df_test['word_seg'])
y_train = df_train['class']-1

"""训练决策树分类器"""
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)

"""根据上面训练好的分类器对测试集的每个样本进行预测"""
y_test = classifier.predict(x_test)

"""将测试集的预测结果保存至本地"""
df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:, ['id', 'class']]
df_result.to_csv('../results/beginner.csv', index=False)

print("完成...............")

