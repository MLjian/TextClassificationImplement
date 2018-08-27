"""
1 简介  ：根据官方给的数据集中'的word_seg'内容，训练词向量，生成word_idx_dict和vectors_arr两个结果，并保存
2 注意  ：1）需要16g内存的电脑，否则由于数据量大，会导致内存溢出。（解决方案：可通过迭代器的格式读入数据。
            见https://rare-technologies.com/word2vec-tutorial/#online_training__resuming）
"""

import pandas as pd
import gensim
import time
import pickle
import numpy as np

vector_size = 100


#=======================================================================================================================
# 0 辅助函数
#=======================================================================================================================

def sentence2list(sentence):
    return sentence.strip().split()

start_time = time.time()


#=======================================================================================================================
# 1 准备训练数据
#=======================================================================================================================

print("准备数据................\n")
df_train = pd.read_csv('../data/data_ori/train_set.csv')
df_test = pd.read_csv('../data/data_ori/test_set.csv')
sentences_train = list(df_train.loc[:, 'word_seg'].apply(sentence2list))
sentences_test = list(df_test.loc[:, 'word_seg'].apply(sentence2list))
sentences = sentences_train + sentences_test
print("准备数据完成!\n")

#=======================================================================================================================
# 2 训练
#=======================================================================================================================
print("开始训练................\n")
model = gensim.models.Word2Vec(sentences=sentences, size=vector_size, window=5, min_count=5, workers=8, sg=0, iter=5)
print("训练完成!\n")

#=======================================================================================================================
# 3 提取词汇表及vectors,并保存
#=======================================================================================================================
print(" 保存训练结果...........\n")
wv = model.wv
vocab_list = wv.index2word

word_idx_dict = {}
for idx, word in enumerate(vocab_list):
    word_idx_dict[word] = idx
    
vectors_arr = wv.vectors
vectors_arr = np.concatenate((np.zeros(vector_size)[np.newaxis, :], vectors_arr), axis=0)#第0位置的vector为'unk'的vector

f_wordidx = open('./word_seg_word_idx_dict.pkl', 'wb')
f_vectors = open('./word_seg_vectors_arr.pkl', 'wb')
pickle.dump(word_idx_dict, f_wordidx)
pickle.dump(vectors_arr, f_vectors)
f_wordidx.close()
f_vectors.close()
print("训练结果已保存到该目录下！\n")

end_time = time.time()
print("耗时：{}s\n".format(end_time - start_time))