"""
@简介 ：
        对train_set.csv：删除'article'列；对classify - 1；截断；word2vec；增加一列'length'；划分训练集和验证集；以df形式保存至本地；
        对test_set.csv：删除'article'列；截断；word2vec；按句子长度降序排列；保存至本地；
@使用说明 ：
        调整参数后运行；
"""
import pandas as pd
import pickle
import time
from sklearn.model_selection import train_test_split

"""
========================================================================================================================
0 调整参数
========================================================================================================================
"""
train = True                    #处理训练集还是测试集
max_length = 2000               #句子截断后的最大长度
time_start = time.time()

"""
========================================================================================================================
1 加载数据
========================================================================================================================
"""
if train:
    data_path = './data_ori/train_set.csv'
else:
    data_path = './data_ori/test_set.csv'

df_ori = pd.read_csv(data_path)

df_ori.drop(columns='article', inplace=True)
if train:
    df_ori['classify'] = df_ori['classify'] - 1

"""加载word2index的dict"""
f_word2index = open('../word2vec/word_seg_word_idx_dict.pkl', 'rb')
word_index_dict = pickle.load(f_word2index)

"""
========================================================================================================================
2 辅助函数（由于代码原因，辅助函数需要位于代码的前面）
========================================================================================================================
"""


def word2index(word):
    """将一个word转换成"""
    if word in word_index_dict:
        return word_index_dict[word]
    else:
        return 0


def sentence2index(sentence):
    """将一个句子转换成index的list，并截断"""
    sentence_list = sentence.strip().split()[:max_length]
    list_index = list(map(word2index, sentence_list))
    return list_index

"""
========================================================================================================================
3 word2index
========================================================================================================================
"""
df_ori['word_seg'] = df_ori['word_seg'].apply(sentence2index)


"""
========================================================================================================================
4 按类别进行分层采样，划分训练集和验证集
========================================================================================================================
"""
if train:
    df_x = df_ori.loc[:, 'word_seg']
    df_y = df_ori.loc[:, 'classify']
    x_train, x_vali, y_train, y_vali = train_test_split(df_x, df_y, test_size=0.1, random_state=0)
    df_train = pd.concat((x_train, y_train), axis=1)
    df_vali = pd.concat((x_vali, y_vali), axis=1)

"""
========================================================================================================================
5 增加一列‘length', 并保存处理后的结果到硬盘
========================================================================================================================
"""
if train:
    df_train['length'] = df_train['word_seg'].apply(len)
    df_vali['length'] = df_vali['word_seg'].apply(len)
    df_train.sort_values(by='length', ascending=False, inplace=True)
    df_vali.sort_values(by='length', ascending=False, inplace=True)
    f_train = open('./data_pro/df_train.pkl', 'wb')
    f_vali = open('./data_pro/df_vali.pkl', 'wb')
    pickle.dump(df_train, f_train)
    f_train.close()
    pickle.dump(df_vali, f_vali)
    f_vali.close()
else:
    f_test = open('./data_pro/df_test.pkl', 'wb')
    df_ori['length'] = df_ori['word_seg'].apply(len)
    df_ori.sort_values(by='length', ascending=False, inplace=True)
    df_ori.drop(columns='length', inplace=True)
    pickle.dump(df_ori, f_test)
    f_test.close()

time_end = time.time()
print("耗时：{} s".format(time_end - time_start))


