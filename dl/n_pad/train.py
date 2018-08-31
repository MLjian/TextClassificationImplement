import pandas as pd
import pickle
import torch
import torch.nn as nn
from models.lstm_sum import LSTMsum
from config import opt
import visdom
import pdb
import time
from torch.nn.utils import rnn

#=======================================================================================================================
# 1 加载数据集
#=======================================================================================================================
f_train = open(opt.data_train_path, 'rb')
f_vali = open(opt.data_vali_path, 'rb')
df_train = pickle.load(f_train)
df_vali = pickle.load(f_vali)
f_train.close()
f_vali.close()

#=======================================================================================================================
# 2 定义网络结构；定义损失函数；定义优化器
#=======================================================================================================================
"""定义网络结构"""
f_emb = open(opt.emb_vectors_path, 'rb')
emb_arr = torch.FloatTensor(pickle.load(f_emb))
f_emb.close()
model = LSTMsum(emb_arr=emb_arr, emb_freeze=opt.emb_freeze, input_size=opt.input_size, hidden_size=opt.hidden_size,
                num_layers=opt.num_layers, bidir=opt.bidir, dropout=opt.dropout_lstm, l1_size=opt.l1_size,
                l2_size=opt.l2_size, num_classes=opt.num_classes)
if opt.use_gpu:
    model.cuda()

"""定义损失函数"""
criterion = nn.CrossEntropyLoss()

"""定义优化器"""
if opt.emb_freeze:
    model_paras = filter(lambda p: p.requires_grad, model.parameters())
else:
    model_paras = model.parameters()
optimizer = torch.optim.Adam(params=model_paras, lr=opt.lr, weight_decay=opt.weight_decay)

#=======================================================================================================================
# 3 定义评估函数
#=======================================================================================================================
def eval(model, df_vali, gap_size, use_gpu):

    model.eval()

    acc_sum = 0
    ii = 0
    start_vali = 0
    while start_vali + gap_size <= len(df_vali):
        """取一batch数据"""
        end_vali = start_vali + gap_size
        df_vali_batch = df_vali.iloc[start_vali:end_vali]
        start_vali = end_vali

        """数据预处理：按句子长度降序排序;pad_sequence"""
        df_vali_batch.sort_values(by='length', ascending=False, inplace=True)
        vali_x_batch = list(df_vali_batch.loc[:, 'word_seg'])
        vali_y_batch = list(df_vali_batch.loc[:, 'classify'])
        x_vali_lengths = list(df_vali_batch.loc[:, 'length'])
        vali_x_batch = [torch.Tensor(x_tmp) for x_tmp in vali_x_batch]
        vali_x_batch = rnn.pad_sequence(vali_x_batch, batch_first=True).long()
        x_vali = torch.LongTensor(vali_x_batch)
        y_vali = torch.LongTensor(vali_y_batch)
        #pdb.set_trace()
        if use_gpu:
            x_vali = x_vali.cuda()
            y_vali = y_vali.cuda()

        """前向传播"""
        output_vali = model(x_vali, x_vali_lengths)
        acc_batch = torch.mean(torch.eq(torch.max(output_vali, 1)[1], y_vali).float())
        acc_sum += acc_batch
        ii += 1
    acc_mean = acc_sum / (ii + 1)

    model.train()
    return acc_mean

#=======================================================================================================================
# 4 训练
#=======================================================================================================================
if __name__ == '__main__':
    print("开始训练......................")
    #vis = visdom.Visdom(env=opt.vis_name)
    print_losses = []
    j = 0
    for epoch in range(opt.num_epochs):
        time_start = time.time()
        """shuffle数据，并to_list"""
        df_train = df_train.sample(frac=1)
        start_idx = 0
        while start_idx + opt.batch_size <= len(df_train):
            """取一batch数据"""
            end_idx = start_idx + opt.batch_size
            df_train_batch = df_train.iloc[start_idx:end_idx]
            start_idx = end_idx

            """数据预处理：按句子长度降序排序;pad_sequence"""
            df_train_batch.sort_values(by='length', ascending=False, inplace=True)
            train_x_batch = list(df_train_batch.loc[:, 'word_seg'])
            train_y_batch = list(df_train_batch.loc[:, 'classify'])
            x_lengths = list(df_train_batch.loc[:, 'length'])
            train_x_batch = [torch.Tensor(x_tmp) for x_tmp in train_x_batch]
            train_x_batch = rnn.pad_sequence(train_x_batch, batch_first=True).long()
            x = torch.LongTensor(train_x_batch)
            y = torch.LongTensor(train_y_batch)
            if opt.use_gpu:
                x = x.cuda()
                y = y.cuda()

            """前向传播"""
            output = model(x, x_lengths)

            """BP"""
            loss = criterion(output, y)
            print_losses.append(loss)
            optimizer.zero_grad()
            loss.backward()

            """更新参数"""
            optimizer.step()

            """可视化Loss"""
            j += 1
            if j % opt.print_fre == 0:
                loss_mean = torch.mean(torch.Tensor(print_losses))
                print_losses = []
                print("train_loss : {}".format(loss_mean))
                #vis.line(X=torch.tensor([j]), Y=torch.tensor([loss_mean.item()]), win='loss',
                         #update='append' if j != opt.print_fre else None, opts=dict(xlabel='step', ylabel='loss'))

        """可视化验证集的准确率"""
        acc_vali = eval(model, df_vali, opt.batch_size, opt.use_gpu)
        time_end = time.time()
        print("第 {} epoch, 验证集的准确率：{}, 耗时：{}s".format((epoch + 1), acc_vali, (time_end - time_start)))
        """
        vis.line(X=torch.tensor([epoch + 1]), Y=torch.tensor([acc_vali]), win='acc_vali',
                 update='append' if epoch != 1 else None, opts=dict(xlabel='epoch', ylabel='vali_accuracy'))
        vis.text(
            text='model_name = {}；emb_freeze = {}；input_size = {}； hidden_size = {}； l1_size = {}； l2_size = {}； '
                 'num_layers = {}；lr = {}；weight_decay = {}；data_shuffle = {}；batch_size = {}'.format(
                opt.model_name, opt.emb_freeze, opt.input_size, opt.hidden_size, opt.l1_size, opt.l2_size,
                opt.num_layers, opt.lr, opt.weight_decay, opt.data_shuffle, opt.batch_size), win='paras')
        """
        
        """每epoch,保存已经训练的模型"""
        TrainedModel_path = './trained_models/%d' % (epoch + 1) + '_' + '%f' % acc_vali + '.pkl'
        torch.save(model, TrainedModel_path)
        #vis.save([opt.vis_name])







