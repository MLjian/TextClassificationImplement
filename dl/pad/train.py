"""
@简介 ：训练
"""
from train_cfg import opt
import pickle
from models.lstm_sum import LSTMsum
import torch.nn as nn
import torch
import numpy as np
import time
import visdom
#=======================================================================================================================
# 1 加载数据集
#=======================================================================================================================
f_data = open(opt.data_train_path, 'rb')
mat_train, mat_vali = pickle.load(f_data)
f_data.close()


#=======================================================================================================================
# 2 定义模型/loss/优化器
#=======================================================================================================================
f_vectors = open(opt.word_vector_path, 'rb')
emb_vectors = torch.Tensor(pickle.load(f_vectors))
model = LSTMsum(emb_weights=emb_vectors, emb_freeze=opt.emb_freeze, input_size=opt.input_size, hidden_size=opt.hidden_size,
                  num_layers=opt.num_layers, l1_size=opt.l1_size, l2_size=opt.l2_size, num_classes=opt.num_classes,
                bidir=opt.bidir, lstm_dropout=opt.lstm_dropout)
if opt.use_gpu:
    model.cuda()

criterion = nn.CrossEntropyLoss()

if opt.emb_freeze:
    model_paras = filter(lambda p : p.requires_grad, model.parameters())
else:
    model_paras = model.parameters()
optimizer = torch.optim.Adam(params=model_paras, lr=opt.lr, weight_decay=opt.weight_decay)

#=======================================================================================================================
# 3 定义评估函数
#=======================================================================================================================

def eval(model, data_vali, gap_size, use_gpu):
    """对data_vali尾部的样本可能会有舍弃的现象"""
    model.eval()
    acc_sum = 0
    ii = 0
    start_vali = 0
    while start_vali + gap_size <= data_vali.shape[0]:
        end_vali = start_vali + gap_size
        tmp_vali = data_vali[start_vali:end_vali]
        start_vali = end_vali

        x_vali = torch.LongTensor(tmp_vali[:,:-1])
        y_vali = torch.LongTensor(tmp_vali[:,-1])
        if use_gpu:
            x_vali = x_vali.cuda()
            y_vali = y_vali.cuda()

        out_tmp = model(x_vali)
        pre_tmp = torch.max(out_tmp, dim=1)[1]
        acc_tmp = torch.mean(torch.eq(pre_tmp, y_vali).float()).item()
        acc_sum += acc_tmp
        ii += 1

    acc_vali = acc_sum / ii
    model.train()
    return acc_vali

#=======================================================================================================================
# 4 训练
#=======================================================================================================================
if __name__ == '__main__':
    print("开始训练......................")
    #vis = visdom.Visdom(env=opt.vis_name)
    losses = []
    j = 0
    for epoch in range(opt.num_epochs):
        time_start = time.time()
        if opt.data_shuffle:
            np.random.shuffle(mat_train)
        start = 0
        while start + opt.batch_size <= mat_train.shape[0]:
            end = start + opt.batch_size
            batch_train = mat_train[start:end]
            start = end

            x_batch = torch.LongTensor(batch_train[:, :-1])
            y_batch = torch.LongTensor(batch_train[:, -1])
            if opt.use_gpu:
                x_batch = x_batch.cuda()
                y_batch = y_batch.cuda()

            output = model(x_batch)

            loss = criterion(output, y_batch)
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            losses.append(loss)
            j += 1

            """可视化loss"""
            if j % opt.print_fre == 0:
                loss_mean = torch.Tensor(losses).mean().item()
                losses = []
                print("loss:{}".format(loss_mean))
                #vis.line(X=torch.tensor([j]), Y=torch.tensor([loss_mean]), win='loss',
                        #update='append' if j != opt.print_fre else None, opts=dict(xlabel='step', ylabel='loss'))
        """评估验证集的准确率"""
        acc_vali = eval(model, mat_vali, opt.batch_size, opt.use_gpu)
        time_end = time.time()
        print("第 {} epoch, 验证集的准确率：{}, 耗时：{}s".format((epoch+1), acc_vali, (time_end - time_start)))
        """
        vis.line(X=torch.tensor([epoch + 1]), Y=torch.tensor([acc_vali]), win='acc_vali',
                 update='append' if epoch != 1 else None, opts=dict(xlabel='epoch', ylabel='vali_accuracy'))
        vis.text(
                text='model_name = {}；emb_freeze = {}；input_size = {}； hidden_size = {}； l1_size = {}； l2_size = {}； '
                     'num_layers = {}；lr = {}；weight_decay = {}；data_shuffle = {}；batch_size = {}'.format(
                    opt.model_name,opt.emb_freeze, opt.input_size, opt.hidden_size, opt.l1_size, opt.l2_size,
                    opt.num_layers, opt.lr, opt.weight_decay, opt.data_shuffle, opt.batch_size), win='paras')
        """
        """每epoch,保存已经训练的模型"""
        TrainedModel_path = './trained_models/%d' % (epoch + 1) + '_' + '%f' % acc_vali + '.pkl'
        torch.save(model, TrainedModel_path)
        #vis.save([opt.vis_name])


