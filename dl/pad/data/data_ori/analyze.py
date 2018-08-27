import pandas as pd

df_train = pd.read_csv('./train_set.csv')
df_test = pd.read_csv('./test_set.csv')
print("官方提供的训练集中的样本数为{}个".format(len(df_train)))
print("官方提供的测试集中的样本数为{}个".format(len(df_test)))

"""检测数据集中是否有含有空值的样本，并删除训练集中的含有空值的样本"""
df_train_nan = df_train[df_train.isnull().values == True]
df_test_nan = df_test[df_test.isnull().values == True]
print("训练集中含有空值的样本的个数：{} ".format(len(df_train_nan)))
print("测试集中含有空值的样本的个数：{} ".format(len(df_test_nan)))

"""查看label的类别"""
classes = set(df_train.loc[:, 'classify'].values.tolist())
print("classes:{}".format(classes))

"""查看句子长度的分布情况"""
print("训练集中的'word_seg'句子长度(含空格)的分布情况如下所示：{}".format(df_train['word_seg'].apply(len).describe()))
print("训练集中的'article'句子长度（含空格）的分布情况如下所示：{}".format(df_train['article'].apply(len).describe()))





