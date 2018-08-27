"""
@简介 ：LSTMsum网络：embedding + lstm + sum + fc
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import rnn
import pdb

class LSTMsum(nn.Module):
    def __init__(self, emb_arr, emb_freeze, input_size, hidden_size, num_layers, bidir, dropout, l1_size, l2_size, num_classes):
        super(LSTMsum, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings=emb_arr, freeze=emb_freeze)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidir,
                            batch_first=True, dropout=dropout)
        if bidir:
            self.l1 = nn.Linear(hidden_size*2, l1_size)
        else:
            self.l1 = nn.Linear(hidden_size, l1_size)
        self.l2 = nn.Linear(l1_size, l2_size)
        self.l3 = nn.Linear(l2_size, num_classes)

    def forward(self, input, input_lengths):
        emb_out = self.embedding(input)
        #pdb.set_trace()
        emb_out = rnn.pack_padded_sequence(emb_out, lengths=input_lengths, batch_first=True)
        lstm_out, _ = self.lstm(emb_out)
        lstm_out, _ = rnn.pad_packed_sequence(lstm_out, batch_first=True)
        sum_out = torch.sum(lstm_out, dim=1)
        l1_out = F.relu(self.l1(sum_out))
        l2_out = F.relu(self.l2(l1_out))
        l3_out = self.l3(l2_out)
        return l3_out
