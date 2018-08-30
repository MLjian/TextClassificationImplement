# _*_ coding: utf-8 _*_
"""
@简介  ： 删除'article'列；对classify-1；word2index；划分训练集和验证集；转换成Numpy格式
@使用说明   ：修改宏变量train，选择预处理训练集还是测试集
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

#=======================================================================================================================
# 0 调整参数
#=======================================================================================================================
train = True
fix_len = 2000
w2i_path = '../word2vec/word_seg_word_idx_dict.pkl'

#=======================================================================================================================
# 1 加载数据集
#=======================================================================================================================
if train:
    data_path = './data_ori/train_set.csv'
else:
    data_path = './data_ori/test_set.csv'

df_ori = pd.read_csv(data_path)
df_ori.drop(columns='article', inplace=True)

"""加载word2index的dict"""
f_word2index = open(w2i_path, 'rb')
word_index_dict = pickle.load(f_word2index)

#=======================================================================================================================
# 2 辅助函数（由于代码原因，辅助函数需要位于代码的前面）
#=======================================================================================================================
def word2index(word):
    """将一个word转换成index"""
    if word in word_index_dict:
        return word_index_dict[word]
    else:
        return 0

def sentence2index(sentence):
    """将一个句子转换成index的list，并截断或补零"""
    word_list = sentence.strip().split()
    index_list = list(map(word2index, word_list))
    len_sen = len(index_list)
    if  len_sen < fix_len:
        index_list = index_list + [0]*(fix_len - len_sen)
    else:
        index_list = index_list[:fix_len]
    return index_list

#=======================================================================================================================
# 3 删除'article'列；word2index，并截断或补零；
#=======================================================================================================================
if train:
    df_ori.loc[:, 'class'] = df_ori.loc[:, 'class'] - 1
df_ori.loc[:, 'word_seg'] = df_ori.loc[:, 'word_seg'].apply(sentence2index)

#=======================================================================================================================
# 4 划分数据集并转换成ndarray，并存储到本地
#=======================================================================================================================
x = np.array(list(df_ori.loc[:, 'word_seg']))
if train:
    y = df_ori.loc[:, 'class'].values
    x_train, x_vali, y_train, y_vali = train_test_split(x, y, test_size=0.07, random_state=0)
    mat_train = np.concatenate((x_train, y_train[:, np.newaxis]), axis=1)
    mat_vali = np.concatenate((x_vali, y_vali[:, np.newaxis]), axis=1)
    data = (mat_train, mat_vali)
else:
    data = x

if train:
    f_data = open('./data_pro/data_train.pkl', 'wb')
else:
    f_data = open('./data_pro/data_test.pkl', 'wb')
pickle.dump(data, f_data)
f_data.close()

