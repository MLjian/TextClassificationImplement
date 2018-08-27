"""
@简介 ：网络模型及训练的参数的配置文件，方便程序的调试/修改/迁移
@使用 ：需要import即可
"""


class DefaultConfig(object):
    """数据集相关参数"""
    data_train_path = './data/data_pro/df_train.pkl'
    data_vali_path = './data/data_pro/df_vali.pkl'
    emb_vectors_path = './word2vec/word_seg_vectors_arr.pkl'

    """模型结构参数"""
    model_name = 'LSTMsum'
    bidir = True
    emb_freeze = True
    input_size = 100
    hidden_size = 400
    l1_size = 400
    l2_size = 200
    num_classes = 20
    num_layers = 3
    dropout_lstm = 0

    """训练参数"""
    use_gpu = True
    lr = 0.001
    weight_decay = 0
    vis_name = 'n_p_LSTMsum'
    num_epochs = 30
    data_shuffle = True
    batch_size = 32
    print_fre = 400

opt = DefaultConfig()
