使用流程：
1）先将原始数据集下载至data/data_ori
2)  运行data/data_ori/analyze.py, 对数据进行简单分析；
3）运行word2vec/train_word2vec.py训练词向量；
4）运行data/data_process.py ,对数据进行预处理；
5）配置train_cfg文件进行参数的配置，并保存；
6）运行train.py进行训练；

说明:
1) [models]保存我们所使用的网络模型；
2）[trained_models]保存我们已经训练好的网络模型；