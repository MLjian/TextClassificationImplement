class DefaultConfig(object):
    """数据集相关参数"""
    data_train_path = './data/data_pro/data_train.pkl'
    word_vector_path = './word2vec/word_seg_vectors_arr.pkl'

    """模型结构参数"""
    model_name = 'LSTMsum'
    emb_freeze = True
    input_size = 100
    hidden_size = 400
    bidir = True
    num_layers = 3
    lstm_dropout = 0
    l1_size = 400
    l2_size = 200
    num_classes = 20

    """训练参数"""
    use_gpu = True
    lr = 0.001
    weight_decay = 0
    vis_name = 'd_LSTMsum'
    num_epochs = 30
    data_shuffle = True
    batch_size = 16
    print_fre = 400

opt = DefaultConfig()